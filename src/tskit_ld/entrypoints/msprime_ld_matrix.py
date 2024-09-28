import json
from collections.abc import Iterator
from itertools import groupby
from pathlib import Path
from typing import cast

import numcodecs
import numpy as np
import pydantic
import structlog
import tskit
import zarr
from htcluster.api.job import BaseModel, job_wrapper, log_config
from more_itertools import zip_equal

from tskit_ld.types import NPFloat64Array

from .common.msprime import SimulationParams, run_msprime

log_config()
LOG = structlog.get_logger()

COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=9, shuffle=numcodecs.Blosc.SHUFFLE)
DS_KW = {"chunks": True, "compressor": COMPRESSOR, "cache_metadata": False}


class LDMatrixParams(BaseModel):
    sample_sets: list[list[int]] | None = None
    sites: list[list[int]] | None = None
    positions: list[list[float | int]] | None = None
    summarize_site_by_tree: bool = False
    merge_mean: bool = False
    stat: list[str] | str
    mode: list[str] | str

    @pydantic.field_validator("mode")
    @classmethod
    def validate_mode(cls, v: list[str] | str) -> list[str] | str:
        match v:
            case str():  # parsing from json
                assert v in {"site", "branch"}, f"invalid mode specified: {v}"
            case list():  # from python
                assert (
                    len(diff := set(v) ^ {"site", "branch"}) == 0
                ), f"invalid mode(s) specified: {diff}"
        return v


class MsprimeLDMatrixParams(BaseModel):
    sim: SimulationParams
    ld_matrix: LDMatrixParams


class JobParams(BaseModel):
    in_files: None
    out_files: Path
    params: MsprimeLDMatrixParams


def compute_ld_matrix(
    ts: tskit.TreeSequence, params: LDMatrixParams, mode: str
) -> Iterator[tuple[str, NPFloat64Array]]:
    stats = [params.stat] if isinstance(params.stat, str) else params.stat
    if mode == "branch":
        for stat in stats:
            LOG.info(mode, stat=stat)
            yield (
                stat,
                ts.ld_matrix(
                    sample_sets=params.sample_sets,
                    positions=params.positions,
                    mode=mode,
                    stat=stat,
                ),
            )
    elif mode == "site":
        for stat in stats:
            LOG.info(mode, stat=stat)
            yield (
                stat,
                ts.ld_matrix(
                    sample_sets=params.sample_sets,
                    sites=params.sites,
                    mode=mode,
                    stat=stat,
                ),
            )
    else:
        raise ValueError(f"Invalid mode {mode}")


def get_n_reps(params: SimulationParams) -> tuple[int, int]:
    n_anc_reps = 1 or params.ancestry_params.num_replicates
    n_mut_reps = 0
    if (mut_params := params.mutation_params) is not None:
        if (mut_seeds := mut_params.random_seeds) is not None:
            n_mut_reps = len(mut_seeds)
        else:
            n_mut_reps = 1
    return n_anc_reps, n_mut_reps


def add_job_metadata(g: zarr.Group, params: MsprimeLDMatrixParams) -> None:
    attrs = {
        "stats": params.ld_matrix.stat,
        "anc_seed": params.sim.ancestry_params.random_seed,
        "anc_n_reps": params.sim.ancestry_params.num_replicates,
    }
    if params.sim.mutation_params is not None:
        attrs["mut_seeds"] = params.sim.mutation_params.random_seeds
    g.attrs.update(**attrs)


def add_sim_metadata(g: zarr.Group, ts: tskit.TreeSequence) -> None:
    g.attrs.update(
        **{
            "provenance": [json.loads(p.record) for p in ts.provenances()],
            "tree_stats": {
                "num_edges": ts.num_edges,
                "num_trees": ts.num_trees,
                "num_mutations": ts.num_mutations,
                "num_sites": ts.num_sites,
            },
        }
    )


def summarize_site_ld_by_tree(
    ts: tskit.TreeSequence, ld: NPFloat64Array
) -> NPFloat64Array:
    # simplify things by masking the lower triangle and diagonal
    ld[np.tril_indices_from(ld)] = np.nan
    # sites grouped by tree boundaries
    tree_groups = (
        np.searchsorted(
            cast(NPFloat64Array, ts.breakpoints(as_array=True)),
            ts.sites_position,
            side="right",
        )
        - 1
    )
    # get site breakpoint indexes for the tree groups
    g, idx = np.unique(tree_groups, return_index=True)
    # we store the mean to be summarized during analysis
    stat_mean = np.zeros((ts.num_trees, ts.num_trees), dtype=np.float64)
    # iterate over the upper triangle chunks, summing all non-nan elements
    # NB: we actually cross the upper triangle when considering diagonal chunks
    #     but this is mitigated by nan-masking the lower triangle
    for inner, (i, r) in enumerate(zip_equal(g, np.vsplit(ld, idx[1:]))):
        for j, c in zip_equal(g[inner:], np.hsplit(r, idx[inner + 1 :])):
            stat_mean[i, j] = np.nanmean(c)
    return stat_mean


def merge_result(out_path: Path) -> None:
    """
    Merge data by taking the mean of each replicate, replace the final output

    NB Does not retain provenance metadata
    TODO: only works for sites
    """
    merged_path = Path(out_path).with_suffix(".merged")
    with zarr.ZipStore(out_path, mode="r") as store, zarr.ZipStore(
        merged_path, mode="w"
    ) as merged_store:
        merged_root = zarr.group(store=merged_store)
        root = zarr.group(store=store)
        merged_root.attrs[out_path] = root.attrs.asdict()
        mg = merged_root.create_group(out_path)
        for stat in root.attrs["stats"]:
            mean = np.nanmean(
                np.dstack([g["site"][stat] for g in root.values()]), axis=2
            )
            shape = {g["site"][stat].shape for g in root.values()}
            if {mean.shape} != shape:
                raise ValueError(f"shapes disagree: {mean.shape}, {shape}")
            mg.create_dataset(
                stat, data=mean, shape=mean.shape, dtype=mean.dtype, **DS_KW
            )
    # Replace the out path with the merged data
    merged_path.rename(out_path)


@job_wrapper(JobParams)
def main(args: JobParams) -> None:
    params = args.params
    n_anc_reps, n_mut_reps = get_n_reps(params.sim)
    n_reps = n_anc_reps * n_mut_reps
    ld_params = params.ld_matrix
    LOG.info(
        "Starting",
        n_anc_reps=n_anc_reps,
        n_mut_reps=n_mut_reps,
        n_tot_reps=n_reps,
        stats=ld_params.stat,
    )
    match ld_params.mode:
        case str():
            modes = set([ld_params.mode])
        case list():
            modes = set(ld_params.mode)

    rep = 1
    with zarr.ZipStore(args.out_files, mode="w") as store:
        root = zarr.group(store=store)
        add_job_metadata(root, params)

        for _, group in groupby(run_msprime(params.sim), key=lambda g: g[0]):
            compute_branch = "branch" in modes
            for anc_rep, mut_rep, ts in group:
                LOG.info("Computing LD", rep=f"{rep}/{n_reps}")
                rep_group = root.create_group((anc_rep, mut_rep))
                add_sim_metadata(rep_group, ts)
                if compute_branch:
                    g = rep_group.create_group("branch")
                    for stat, ld in compute_ld_matrix(ts, ld_params, "branch"):
                        LOG.info("writing matrix")
                        g.create_dataset(
                            stat, data=ld, shape=ld.shape, dtype=ld.dtype, **DS_KW
                        )

                    LOG.info("writing breakpoints")
                    bp = cast(NPFloat64Array, ts.breakpoints(as_array=True))
                    g.create_dataset(
                        "breakpoints", data=bp, shape=bp.shape, dtype=bp.dtype, **DS_KW
                    )
                    LOG.info("writing breakpoints")
                    tot_branch_len = np.array(
                        [t.total_branch_length for t in ts.trees()]
                    )
                    g.create_dataset(
                        "total_branch_length",
                        data=tot_branch_len,
                        shape=tot_branch_len.shape,
                        dtype=tot_branch_len.dtype,
                        **DS_KW,
                    )
                    compute_branch = False
                if "site" in modes:
                    g = rep_group.create_group("site")
                    for stat, ld in compute_ld_matrix(ts, ld_params, "site"):
                        if ld_params.summarize_site_by_tree:
                            LOG.info("writing summarized LD data")
                            ld = summarize_site_ld_by_tree(ts, ld)
                        else:
                            LOG.info("writing matrix")
                        g.create_dataset(
                            stat, data=ld, shape=ld.shape, dtype=ld.dtype, **DS_KW
                        )

                    LOG.info("writing positions")
                    pos = ts.sites_position
                    g.create_dataset(
                        "sites_pos", data=pos, shape=pos.shape, dtype=pos.dtype, **DS_KW
                    )
                rep += 1

    if params.ld_matrix.merge_mean:
        merge_result(args.out_files)

    LOG.info("wrote result", file=args.out_files)
