import json
from itertools import product
from pathlib import Path
from typing import Generator

import demes
import msprime
import numpy as np
import polars as pl
import structlog
import tskit
from htcluster.api.job import BaseModel, job_wrapper, log_config
from more_itertools import zip_equal

from tskit_ld.io import write_parquet
from tskit_ld.ld_decay import DecayReturnType, ld_decay, ld_decay_two_way

log_config()
LOG = structlog.get_logger()

type OneWayDecayJobReturnType = Generator[
    # rep, stat, samp, time, decay
    tuple[int, list[tuple[str, str, int, DecayReturnType]]], None, None
]
type TwoWayDecayJobReturnType = Generator[
    # rep, stat, samp_a, samp_b, time_a, time_b, decay
    tuple[int, list[tuple[str, str, str, int, int, DecayReturnType]]], None, None
]


# a bit hacky, but works
class MigSeed(BaseModel):
    migration_rate: float
    random_seed: int


class MsprimeSimParams(BaseModel):
    migseed: MigSeed
    migrations: dict[str, list[str]]
    sample_times: list[float]
    sequence_length: float
    mutation_rate: float
    recombination_rate: float
    sample_size: int
    demography: dict
    n_reps: int


class OneWayDecayParams(BaseModel):
    sample_time_indices: list[int]
    sample_groups: list[str]
    stats: list[str]


class TwoWayDecayParams(BaseModel):
    a_sample_groups: list[str]
    b_sample_groups: list[str]
    stats: list[str]


class DecayArgs(BaseModel):
    n_cpus: int
    chunk_size: int
    bins: list[float]
    max_dist: int  # TODO: could be float too
    one_way: OneWayDecayParams
    two_way: TwoWayDecayParams


class DecayParams(BaseModel):
    # TODO: float time?
    sample_times: list[list[int]]
    args: DecayArgs


class MsprimeLDDecayParams(BaseModel):
    sim: MsprimeSimParams
    decay: DecayParams


class JobParams(BaseModel):
    in_files: None
    out_files: Path
    params: MsprimeLDDecayParams


@job_wrapper(JobParams)
def main(args: JobParams) -> None:
    params = args.params

    LOG.info("running msprime")
    tss = run_msprime(params.sim)
    decay, meta = one_way_result_to_df(
        compute_decay_one_way(
            tss,
            params.decay.args.one_way,
            params.decay.sample_times,
            params.decay.args.bins,
            params.decay.args.chunk_size,
            params.decay.args.max_dist,
            params.decay.args.n_cpus,
        )
    )
    decay_two_way, meta_two_way = two_way_result_to_df(
        compute_decay_two_way(
            tss,
            params.decay.args.two_way,
            params.decay.sample_times,
            params.decay.args.bins,
            params.decay.args.chunk_size,
            params.decay.args.max_dist,
            params.decay.args.n_cpus,
        )
    )
    result = pl.concat(
        [decay.rename(lambda s: s + "-1"), decay_two_way.rename(lambda s: s + "-2")],
        how="horizontal",
    )
    result_meta = {
        "meta": json.dumps(meta),
        "columns": json.dumps(["col", "rep", "stat", "name", "time"]),
        "meta_two_way": json.dumps(meta_two_way),
        "columns_two_way": json.dumps(
            ["col", "rep", "stat", "a_name", "b_name", "a_time", "b_time"]
        ),
        "args": args.model_dump_json(),
    }
    write_parquet(result, args.out_files, result_meta)
    LOG.info("wrote result", file=args.out_files)


def preprocess_demography(params: MsprimeSimParams) -> demes.Graph:
    """Convert demography specification to Demes Graph. Since we're specifying
    the migrations and rate in different places (due to ease of parameter
    specification), we need to perform this basic manipulation to obtain a valid
    input to the demes graph.

    :param params: Simulation params containing demography specification
    :returns: Demes graph parsed from final demography specification

    """
    demography = params.demography
    demography.update(
        {
            "migrations": [
                {**params.migrations, **{"rate": params.migseed.migration_rate}}
            ]
        }
    )
    return demes.Graph.fromdict(demography)


# TODO: use .common.msprime.run_msprime
def run_msprime(params: MsprimeSimParams) -> list[tskit.TreeSequence]:
    """Run msprime with provided parameters. Will compute multiple reps if specified

    :param params: Parameters for ancestry and mutation simulations
    :returns: List of tree sequences with mutations

    """
    demog = preprocess_demography(params)
    sample_sets = [
        msprime.SampleSet(params.sample_size, population=p, time=t)
        for p, t in product([d.name for d in demog.demes], params.sample_times)
    ]
    tss = msprime.sim_ancestry(
        samples=sample_sets,
        demography=msprime.Demography.from_demes(demog),
        num_replicates=params.n_reps,
        sequence_length=params.sequence_length,
        recombination_rate=params.recombination_rate,
        random_seed=params.migseed.random_seed,
    )
    # TODO: we reuse the seed for each rep here.
    return [
        msprime.sim_mutations(
            ts, rate=params.mutation_rate, random_seed=params.migseed.random_seed
        )
        for ts in tss
    ]


def compute_decay_one_way(
    tss: list[tskit.TreeSequence],
    params: OneWayDecayParams,
    sample_times: list[list[int]],
    bins: list[float],
    chunk_size: int,
    max_dist: int,
    n_cpus: int,
) -> OneWayDecayJobReturnType:
    """Compute LD decay within one sample set

    :param tss: List of tree sequences, one item per replicate
    :param params: Parameters for LD decay
    :param bins: Bins to compute LD decay over
    :param chunk_size: Compute LD matricies with this chunk size
    :param max_dist: Maximum distance over which to compute LD decay
    :param n_cpus: Number of CPUs to use when computing LD decay
    :returns: Generator of results, one item per replicate. Results are lists
              with sample sets and the metadata associated with the LD decay
              computation, including samples compared and the stat computed.

    """
    for (rep, ts), stat in product(enumerate(tss), params.stats):
        pop_name_to_id = {p.metadata["name"]: p.id for p in ts.populations()}
        sample_sets = [
            (name, t, ts.samples(pop_name_to_id[name], time=t))
            for name, idx in zip_equal(params.sample_groups, params.sample_time_indices)
            for t in sample_times[idx]
        ]
        out = []
        for name, time, ss in sample_sets:
            LOG.info(
                "computing LD decay one way", rep=rep, stat=stat, name=name, time=time
            )
            out.append(
                (
                    stat,
                    name,
                    time,
                    ld_decay(
                        ts,
                        bins=np.asarray(bins),
                        n_threads=n_cpus,
                        stat=stat,
                        chunk_size=chunk_size,
                        max_dist=max_dist,
                        sample_sets=[ss],
                    ),
                )
            )
        yield rep, out


def compute_decay_two_way(
    tss: list[tskit.TreeSequence],
    params: TwoWayDecayParams,
    sample_times: list[list[int]],
    bins: list[float],
    chunk_size: int,
    max_dist: int,
    n_cpus: int,
) -> TwoWayDecayJobReturnType:
    """Compute two-way LD decay between two sample sets

    :param tss: List of tree sequences, one item per replicate
    :param params: Parameters for LD decay
    :param bins: Bins to compute LD decay over
    :param chunk_size: Compute LD matricies with this chunk size
    :param max_dist: Maximum distance over which to compute LD decay
    :param n_cpus: Number of CPUs to use when computing LD decay
    :returns: Generator of results, one item per replicate. Results are lists
              with sample sets and the metadata associated with the LD decay
              computation, including samples compared and the stat computed.

    """
    for (rep, ts), stat in product(enumerate(tss), params.stats):
        pop_name_to_id = {p.metadata["name"]: p.id for p in ts.populations()}

        a_sample_sets = []
        b_sample_sets = []
        for a_group, b_group in zip(params.a_sample_groups, params.b_sample_groups):
            for a_time, b_time in zip(*sample_times):
                # We can't compute two_way unbiased stats on sample sets that aren't
                # disjoint. We rely on the one_way stats for these results. Ideally,
                # the params wouldn't give us these combinations, but I don't have
                # time to rework those.
                if (a_group == b_group) and (a_time == b_time):
                    LOG.info(
                        "skipping two_way group",
                        a_group=a_group,
                        a_time=a_time,
                        b_group=b_group,
                        b_time=b_time,
                    )
                    continue
                a_sample_sets.append(
                    (a_group, a_time, ts.samples(pop_name_to_id[a_group], time=a_time))
                )
                b_sample_sets.append(
                    (b_group, b_time, ts.samples(pop_name_to_id[b_group], time=b_time))
                )

        assert len(a_sample_sets) == len(
            b_sample_sets
        ), "a_sample_sets and b_sample_sets must be of equal length"
        out = []
        for (a_name, a_time, a_ss), (b_name, b_time, b_ss) in zip_equal(
            a_sample_sets, b_sample_sets
        ):
            LOG.info(
                "computing LD decay two way",
                rep=rep,
                stat=stat,
                a_name=a_name,
                a_time=a_time,
                b_name=b_name,
                b_time=b_time,
            )
            out.append(
                (
                    stat,
                    a_name,
                    b_name,
                    a_time,
                    b_time,
                    ld_decay_two_way(
                        ts,
                        bins=np.asarray(bins),
                        n_threads=n_cpus,
                        stat=stat,
                        chunk_size=chunk_size,
                        max_dist=max_dist,
                        sample_sets=[a_ss, b_ss],
                    ),
                )
            )
        yield rep, out


def one_way_result_to_df(
    results: OneWayDecayJobReturnType,
) -> tuple[pl.DataFrame, list]:
    """Convert LD decay results to a polars dataframe and record column
    metadata to be saved to disk

    :param results: LD decay results
    :returns: Tuple of dataframe and list describing the columns (1 row per col)

    """
    out = dict()
    meta = list()
    i = 0
    last_bins = None
    for rep, result in results:
        for stat, name, time, (b, c, d) in result:
            if last_bins is not None:
                assert (
                    b == last_bins
                ).all(), f"bins not equal b: {b} last: {last_bins}"
            last_bins = b.copy()
            for k, v in [
                (f"counts_{i}", c),
                (f"decay_{i}", d),
            ]:
                out[k] = v
            meta.append((i, rep, stat, name, time))
            i += 1
    return pl.DataFrame(out), meta


def two_way_result_to_df(
    results: TwoWayDecayJobReturnType,
) -> tuple[pl.DataFrame, list]:
    """Convert two-way decay results to a polars dataframe and record column
    metadata to be saved to disk

    :param results: Two-way LD decay results
    :returns: Tuple of dataframe and list describing the columns (1 row per col)

    """
    out = dict()
    meta = list()
    i = 0
    last_bins = None
    for rep, result in results:
        for stat, a_name, b_name, a_time, b_time, (b, c, d) in result:
            if last_bins is not None:
                assert (
                    b == last_bins
                ).all(), f"bins not equal b: {b} last: {last_bins}"
            last_bins = b.copy()
            for k, v in [
                (f"counts_{i}", c),
                (f"decay_{i}", d),
            ]:
                out[k] = v
            meta.append((i, rep, stat, a_name, b_name, a_time, b_time))
            i += 1
    return pl.DataFrame(out), meta
