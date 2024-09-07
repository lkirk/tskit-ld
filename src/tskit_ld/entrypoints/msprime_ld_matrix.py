from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pydantic
import structlog
import tskit
from htcluster.api import BaseModel, job_wrapper, log_config

from .common.msprime import SimulationParams, run_msprime

log_config()
LOG = structlog.get_logger()


class LDMatrixParams(BaseModel):
    sample_sets: list[list[int]] | None
    sites: list[list[int]] | None
    positions: list[list[float | int]] | None
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


def get_n_reps(params: SimulationParams) -> tuple[int, int]:
    n_anc_reps = 1 or params.ancestry_params.num_replicates
    n_mut_reps = 0
    if (mut_params := params.mutation_params) is not None:
        if (mut_seeds := mut_params.random_seeds) is not None:
            n_mut_reps = len(mut_seeds)
        else:
            n_mut_reps = 1
    return n_anc_reps, n_mut_reps


@job_wrapper(JobParams)
def main(args: JobParams) -> None:
    params = args.params
    ld_site_results = []
    ld_branch_results = []
    branch_stats = []
    site_stats = []
    provenance = []
    tree_stats = []
    reps = []
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

    last_anc = 0
    for rep, (anc_rep, ts) in enumerate(run_msprime(params.sim)):
        reps.append((rep, anc_rep))
        LOG.info(
            "Computing LD Matrix", rep=f"{rep}/{n_reps}", stats=params.ld_matrix.stat
        )
        if "branch" in modes and last_anc != anc_rep:
            # only compute branch stats if we're using a new ancestry simulation
            last_anc = anc_rep
            for stat, ld in compute_ld_matrix(ts, params.ld_matrix, "branch"):
                ld_branch_results.append(ld)
                branch_stats.append(stat)

        if "site" in modes:
            for stat, ld in compute_ld_matrix(ts, params.ld_matrix, "site"):
                ld_site_results.append(ld)
                site_stats.append(stat)
        provenance.append([p.record for p in ts.provenances()])
        tree_stats.append([ts.num_edges, ts.num_trees, ts.num_mutations, ts.num_sites])

    ld_site_results = np.stack(ld_site_results)
    np.savez_compressed(
        args.out_files,
        ld_site=ld_site_results,
        site_stats=site_stats,
        ld_branch=ld_branch_results,
        branch_stats=branch_stats,
        provenance=provenance,
        tree_stats=np.array(tree_stats, dtype=np.int64),
    )
    LOG.info("wrote result", file=args.out_files)
