import json
from collections.abc import Iterator
from itertools import groupby
from pathlib import Path
from shutil import rmtree
from typing import Any, cast

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


def store_job_metadata(g: zarr.Group, params: MsprimeLDMatrixParams) -> None:
    attrs = {
        "stats": params.ld_matrix.stat,
        "anc_seed": params.sim.ancestry_params.random_seed,
        "anc_n_reps": params.sim.ancestry_params.num_replicates,
    }
    if params.sim.mutation_params is not None:
        attrs["mut_seeds"] = params.sim.mutation_params.random_seeds
    g.attrs.update(**attrs)


def get_sim_metadata(ts: TreeSequence) -> dict[str, Any]:
    return {
        "provenance": [json.loads(p.record) for p in ts.provenances()],
        "tree_stats": {
            "num_edges": ts.num_edges,
            "num_trees": ts.num_trees,
            "num_mutations": ts.num_mutations,
            "num_sites": ts.num_sites,
        },
    }


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


def store_branch_ld(
    ts: TreeSequence, params: LDMatrixParams, rep_group: zarr.Group
) -> None:
    g = rep_group.create_group("branch")
    for stat, ld in compute_ld_matrix(ts, params, "branch"):
        g.create_dataset(stat, data=ld, shape=ld.shape, dtype=ld.dtype, **DS_KW)


def store_site_ld(
    ts: TreeSequence, params: LDMatrixParams, rep_group: zarr.Group
) -> None:
    g = rep_group.require_group("site")
    for stat, ld in compute_ld_matrix(ts, params, "site"):
        if params.sum_site_by_rep:
            ld = sum_site_ld_by_tree(ts, zero_diag(ld))
        elif params.sum_site_by_tree:
            ld = sum_site_ld_by_tree(ts, ld)

        # When summing ld for a whole anc rep, this returns the running sum
        ds = g.require_dataset(
            stat, shape=ld.shape, dtype=ld.dtype, fill_value=0, **DS_KW
        )
        ds[:] += ld  # type: ignore


@job_wrapper(JobParams)
def main(args: JobParams) -> None:
    params = args.params
    tmp_out = args.out_files.with_suffix(".tmp")
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

    store = zarr.DirectoryStore(tmp_out)
    root = zarr.group(store=store)
    store_job_metadata(root, params)
    ld_group = root.create_group("ld")
    meta_group = root.create_group("ts_meta")

    rep = 1
    for _, group in groupby(run_msprime(params.sim), key=lambda g: g[0]):
        anc_seen = set()
        for anc_rep, mut_rep, ts in group:
            rep_key = anc_rep if ld_params.sum_site_by_rep else (anc_rep, mut_rep)
            rep_group = ld_group.require_group(rep_key)

            if anc_rep not in meta_group:  # store basic info about ts and topo
                ts_data_group = meta_group.create_group(anc_rep)
                if ld_params.store_tree_breakpoints:
                    store_tree_breakpoints(ts, ts_data_group)
                if ld_params.store_total_branch_length:
                    store_total_branch_length(ts, ts_data_group)
                ts_data_group.attrs.update(**get_sim_metadata(ts))

            LOG.info("Computing LD", rep=f"{rep}/{n_reps}")
            if "branch" in modes and anc_rep not in anc_seen:
                store_branch_ld(ts, ld_params, rep_group)

            if "site" in modes:
                store_site_ld(ts, ld_params, rep_group)
            anc_seen.add(anc_rep)
            rep += 1

    LOG.info("Moving to zip store", file=args.out_files)
    with zarr.ZipStore(args.out_files, mode="w") as zs:
        zarr.copy_store(store, zs)

    LOG.info("Removing tmp store", dir=tmp_out)
    rmtree(tmp_out)

    LOG.info("wrote result", file=args.out_files)
