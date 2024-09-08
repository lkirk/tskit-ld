import json
from collections.abc import Iterator
from itertools import groupby
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pydantic
import structlog
import tskit
import zarr
from htcluster.api import BaseModel, job_wrapper, log_config

from .common.msprime import SimulationParams, run_msprime

log_config()
LOG = structlog.get_logger()


class LDMatrixParams(BaseModel):
    sample_sets: list[list[int]] | None = None
    sites: list[list[int]] | None = None
    positions: list[list[float | int]] | None = None
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
) -> Iterator[tuple[str, npt.NDArray[np.float64]]]:
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


def compute_ld(
    group: Iterator[tuple[int, int, tskit.TreeSequence]],
    params: LDMatrixParams,
    root: zarr.Group,
    modes: set[str],
) -> None:
    compute_branch = "branch" in modes
    for anc_rep, mut_rep, ts in group:
        rep_group = root.create_group((anc_rep, mut_rep))
        add_sim_metadata(rep_group, ts)
        if compute_branch:
            g = rep_group.create_group("branch")
            for stat, ld in compute_ld_matrix(ts, params, "branch"):
                g.create_dataset(stat, shape=ld.shape, dtype=ld.dtype, chunks=True)
            compute_branch = False
        if "site" in modes:
            g = rep_group.create_group("site")
            for stat, ld in compute_ld_matrix(ts, params, "site"):
                g.create_dataset(stat, shape=ld.shape, dtype=ld.dtype, chunks=True)


@job_wrapper(JobParams)
def main(args: JobParams) -> None:
    params = args.params
    n_anc_reps, n_mut_reps = get_n_reps(params.sim)
    n_reps = n_anc_reps * n_mut_reps
    LOG.info(
        "Starting", n_anc_reps=n_anc_reps, n_mut_reps=n_mut_reps, n_tot_reps=n_reps
    )
    match params.ld_matrix.mode:
        case str():
            modes = set([params.ld_matrix.mode])
        case list():
            modes = set(params.ld_matrix.mode)

    with zarr.ZipStore(args.out_files, mode="w") as store:
        root = zarr.group(store=store)
        add_job_metadata(root, params)

        anc_sim_groups = groupby(run_msprime(params.sim), key=lambda g: g[0])
        for rep, (_, group) in enumerate(anc_sim_groups, 1):
            LOG.info("Computing LD", rep=f"{rep}/{n_reps}", stats=params.ld_matrix.stat)
            compute_ld(group, params.ld_matrix, root, modes)

    LOG.info("wrote result", file=args.out_files)
