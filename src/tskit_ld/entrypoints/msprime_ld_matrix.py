import json
from collections.abc import Iterator
from itertools import groupby
from pathlib import Path
from typing import cast

import numcodecs
import numpy as np
import pydantic
import structlog
import zarr
from htcluster.api.job import BaseModel, job_wrapper, log_config
from tskit import TreeSequence

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
    stat: list[str] | str
    mode: list[str] | str

    sum_site_by_tree: bool = False
    sum_site_by_rep: bool = False  # Implies sum_site_by_tree
    store_tree_breakpoints: bool = False
    store_total_branch_length: bool = False

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
    ts: TreeSequence, params: LDMatrixParams, mode: str
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


def add_sim_metadata(g: zarr.Group, ts: TreeSequence) -> None:
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


def zero_diag(a):
    """
    Zero out the diagonal of a given matrix.
    Will throw an error if matrix is not square.
    Use the return value or not, a copy is not made.
    """
    a[np.diag_indices_from(a)] = 0
    return a


def sum_site_ld_by_tree(ts: TreeSequence, data: NPFloat64Array) -> NPFloat64Array:
    num_sites = np.array([t.num_sites for t in ts.trees()], dtype=np.int64)
    out_idx = np.where(num_sites)[0]
    add_idx = np.cumsum(num_sites[out_idx[:-1]])

    out = np.zeros((ts.num_trees, ts.num_trees), dtype=np.float64)

    if len(add_idx) > 0:
        if add_idx[0] != 0:
            add_idx = np.insert(add_idx, 0, 0)
        out[np.ix_(out_idx, out_idx)] = np.add.reduceat(
            np.add.reduceat(data, add_idx, axis=0), add_idx, axis=1
        )
    return out


def store_tree_breakpoints(ts: TreeSequence, g: zarr.Group) -> None:
    LOG.info("writing breakpoints")
    bp = cast(NPFloat64Array, ts.breakpoints(as_array=True))
    g.create_dataset("breakpoints", data=bp, shape=bp.shape, dtype=bp.dtype, **DS_KW)


def store_total_branch_length(ts: TreeSequence, g: zarr.Group) -> None:
    LOG.info("writing total branch lengths")
    tot_branch_len = np.array([t.total_branch_length for t in ts.trees()])
    g.create_dataset(
        "total_branch_length",
        data=tot_branch_len,
        shape=tot_branch_len.shape,
        dtype=tot_branch_len.dtype,
        **DS_KW,
    )


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
        ld_group = root.create_group("ld")
        meta_group = root.create_group("ts_meta")

        for _, group in groupby(run_msprime(params.sim), key=lambda g: g[0]):
            anc_seen = set()
            for anc_rep, mut_rep, ts in group:
                rep_sum_first = ld_params.sum_site_by_rep and anc_rep not in anc_seen
                if anc_rep not in anc_seen:
                    ts_data_group = meta_group.create_group(anc_rep)
                    if ld_params.store_tree_breakpoints:
                        store_tree_breakpoints(ts, ts_data_group)
                    if ld_params.store_total_branch_length:
                        store_total_branch_length(ts, ts_data_group)

                LOG.info("Computing LD", rep=f"{rep}/{n_reps}")
                if rep_sum_first:
                    rep_group = ld_group.create_group(anc_rep)
                elif not ld_params.sum_site_by_rep:
                    rep_group = ld_group.create_group((anc_rep, mut_rep))
                else:
                    raise Exception("Rep group not created")

                add_sim_metadata(rep_group, ts)

                if "branch" in modes and anc_rep not in anc_seen:
                    g = rep_group.create_group("branch")
                    for stat, ld in compute_ld_matrix(ts, ld_params, "branch"):
                        LOG.info("writing branch matrix")
                        g.create_dataset(
                            stat, data=ld, shape=ld.shape, dtype=ld.dtype, **DS_KW
                        )

                if "site" in modes:
                    if rep_sum_first or not ld_params.sum_site_by_rep:
                        g = rep_group.create_group("site")
                    else:
                        g = rep_group["site"]
                    assert isinstance(g, zarr.Group)  # mypy
                    for stat, ld in compute_ld_matrix(ts, ld_params, "site"):
                        LOG.info("writing summarized LD data")
                        if ld_params.sum_site_by_rep:
                            ld = sum_site_ld_by_tree(ts, zero_diag(ld))
                        else:
                            ld = sum_site_ld_by_tree(ts, ld)
                        if not ld_params.sum_site_by_rep or rep_sum_first:
                            LOG.info("writing site matrix")
                            g.create_dataset(
                                stat,
                                data=ld,
                                shape=ld.shape,
                                dtype=ld.dtype,
                                **DS_KW,
                            )
                        elif ld_params.sum_site_by_rep:
                            g[stat][:] += ld

                    # LOG.info("writing positions")
                    # pos = ts.sites_position
                    # g.create_dataset(
                    #     "sites_pos", data=pos, shape=pos.shape, dtype=pos.dtype, **DS_KW
                    # )

                anc_seen.add(anc_rep)
                rep += 1

    LOG.info("wrote result", file=args.out_files)
