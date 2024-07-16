import json
from pathlib import Path
from typing import Any, Generator, cast

import msprime
import numpy as np
import polars as pl
import pyslim
import tskit
import tszip
from joblib import Parallel, delayed

from .io import read_parquet_file, write_parquet
from .types import IntoExpr, NPInt32Array, NPInt64Array, NPShapeLike, NPUInt32Array

SEED = 23
RNG = np.random.RandomState(23)


def get_seed(size: NPShapeLike = None) -> int | NPInt64Array:
    """
    Generate random seeds for sampling processes. We seed our RNG to
    obtain some determinism for the pipeline, but generate random seeds
    for each step.

    NB: if the ordering changes between steps, seeds (and results) will change
    """
    return RNG.randint(0, 2**31, size=size)


def get_ts_sampling_times(ts: tskit.TreeSequence, normalize: bool) -> NPUInt32Array:
    """
    Parse the sampling times that we've recorded in the tree sequence.

    NB: these times are in ticks and not generation time units. Because
    of this, we're returning integers and not floats.
    """
    assert all(  # First, validate our assumptions about the data type
        [
            0 <= t < np.iinfo(np.uint32).max
            for t in ts.metadata["SLiM"]["user_metadata"]["sampling_times"]
        ]
    ), "sampling time larger than uint32"
    sampling_times = np.array(
        ts.metadata["SLiM"]["user_metadata"]["sampling_times"], dtype=np.uint32
    )
    if normalize:
        return pyslim.slim_time(ts, sampling_times)
    return sampling_times


def get_inds_sampling_time(ts: tskit.TreeSequence) -> NPUInt32Array:
    """
    Get the time at which individuals are alive, which is when we sampled
    them during our simulation.
    """
    # verify assumptions about time units
    assert (
        ts.metadata["SLiM"]["model_type"] == "nonWF"
        and ts.metadata["SLiM"]["stage"] == "late"
    )
    age_offset = 1  # see above assertion
    # sampling time
    st = get_ts_sampling_times(ts, normalize=True)
    # birth time
    bt = np.repeat(ts.individuals_time, len(st)).reshape(-1, len(st))
    # sample ages at sampling
    age = np.repeat(ts.tables.individuals.metadata_vector("age"), len(st)).reshape(
        -1, len(st)
    )
    return st[np.where((bt >= st) & (bt - age < st + age_offset))[1]]


def preprocess_metadata(meta: dict[str, Any]) -> str:
    """
    Strip out unnecessary metadata, flatten and convert values to bytes.
    We flatten the dict to make it row-like for data table construction.
    """
    assert len(meta["SLiM"]["user_metadata"]["params"]) == 1
    assert all(
        [len(v) == 1 for v in meta["SLiM"]["user_metadata"]["params"][0].values()]
    )
    meta["params"] = {
        k: v[0] for k, v in meta["SLiM"]["user_metadata"]["params"][0].items()
    }
    del meta["SLiM"]["user_metadata"]
    meta = {**meta["SLiM"], **meta["params"]}  # flatten dict
    return json.dumps(meta)


def read_ts_and_process_spatial_data(
    ts_path: Path,
    run_id: str,
    run_ids: pl.Enum,
) -> tuple[dict[str, str], pl.DataFrame]:
    ts = tszip.load(ts_path)
    assert (
        np.array([t.num_roots for t in ts.trees()]) == 1
    ).all(), "not all trees have coalesced"
    sampling_time = get_inds_sampling_time(ts)
    assert (ts.individuals_location[:, 2] == 0).all()
    df = pl.DataFrame(
        {
            "ind": np.arange(ts.num_individuals, dtype=np.int32),
            "x": ts.individuals_location[:, 0],
            "y": ts.individuals_location[:, 1],
            "sampling_time": sampling_time,
            "age": ts.tables.individuals.metadata_vector("age"),
        }
    ).with_columns(run_id=pl.lit(run_id, dtype=run_ids))
    return {run_id: preprocess_metadata(ts.metadata)}, df


def process_raw_ind_data_and_write_parquet(
    in_paths: dict[str, Path],
    out_path: Path,
    n_jobs: int,
    verbose: int = 10,
    **process_kwargs: Any,
) -> None:
    assert out_path.parent.exists(), f"{out_path.parent} does not exist"
    run_ids = pl.Enum(
        sorted(in_paths.keys(), key=lambda k: tuple(map(int, k.split("-"))))
    )
    result_iter = cast(
        Generator[tuple[dict[str, bytes], pl.DataFrame], None, None],
        Parallel(verbose=verbose, n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(read_ts_and_process_spatial_data)(f, run_id, run_ids)
            for run_id, f in in_paths.items()
        ),
    )
    df = pl.DataFrame()
    meta = dict()
    # NB no need to rechunk because we're saving to disk immediately
    for m, data in result_iter:
        df.vstack(data, in_place=True)
        meta.update(m)
    write_parquet(df, out_path, meta, **process_kwargs)


def linspace(
    start: int | float,
    stop: int | float,
    num: int,
    endpoint: bool = True,
    eager: bool = False,
) -> pl.Expr | pl.Series:
    delta = stop - start
    div = (num - 1) if endpoint else num
    step = delta / div
    y = (pl.arange(0, num, eager=eager) * step) + start
    if endpoint:
        # lazy equivalent to y[-1] = stop
        y = y.shift(1).shift(-1, fill_value=stop)
    return y


def linear_transect(
    padding: int | float, square_size: int | float, num_squares: int
) -> pl.Expr:
    """
    Sample along a linear transect in adjacent squares. Parameters
    could be better, but this works well enough for now.
    """
    lower = left = padding
    upper = padding + square_size
    n = num_squares + 1
    right = square_size * n

    bounds = linspace(left, right, n)
    in_bounds = bounds.search_sorted(pl.col("x"), side="left")
    in_bounds = (
        pl.when(
            pl.col("x").is_between(left, right, closed="right"),
            pl.col("y").is_between(lower, upper, closed="right"),
        )
        .then(in_bounds)
        .otherwise(None)
    )

    # return in_bounds
    return pl.arange(0, num_squares).gather(in_bounds - 1)


def spatially_sample_individuals(
    df: pl.LazyFrame, n_ind: int, sample_group_fn: IntoExpr
) -> pl.LazyFrame:
    """Given a dataset with the spatial coordinates, run identifiers, and
    individual identifiers, generate spatial sampling boundaries and sample
    n individuals from within those boundaries.

    :param df: dataframe with columns: ind, run_id, x, y
    :param n_ind: number of individuals to sample from sampling boundaries
    :param sample_group_function: function to create spatial boundaries
    :returns: query plan to spatially sample individuals (not evaluated)

    """
    return (
        df.with_columns(sample_group=sample_group_fn)
        .filter(pl.col("sample_group").is_not_null())
        .group_by("sampling_time", "run_id", "sample_group")
        .agg(ind=pl.col("ind").sample(n_ind, seed=cast(int, get_seed())))
        .explode("ind")
    )


def spatially_sample_individuals_join_data(
    in_path: Path, n_ind: int, sample_group_fn: IntoExpr
) -> pl.DataFrame:
    df = read_parquet_file(in_path)
    # unfortunately, we have to  join this with the original data. The initial
    # semi join reduces the memory overhead for performing a massive join. The
    # query optimizer prevents sampled from being computed twice.
    assert isinstance(df, pl.LazyFrame)  # mypy
    sampled = spatially_sample_individuals(df, n_ind, sample_group_fn)
    filtered = df.select("ind", "x", "y", "age", "run_id").join(
        sampled, how="semi", on=["run_id", "ind"]
    )

    return (
        sampled.join(
            filtered,
            on=["run_id", "ind"],
            validate="1:1",
            how="inner",
            coalesce=True,
        )
        .sort("run_id", "sampling_time", "sample_group", "ind")
        .collect()
    )


def simplify_tree_sequence(
    ts: tskit.TreeSequence, sampled: pl.DataFrame
) -> tuple[tskit.TreeSequence, NPInt32Array]:
    # Nodes to keep. We must keep the nodes from the individual (they're diploid in this case)
    ind_nodes = np.vstack([ts.individual(i).nodes for i in sampled["ind"]])
    sts, node_map = cast(
        tuple[tskit.TreeSequence, NPInt32Array],
        ts.simplify(samples=ind_nodes.reshape(-1), map_nodes=True),
    )
    new_nodes = node_map[ind_nodes]
    s_nodes = sts.nodes_individual[new_nodes]
    assert (new_nodes != -1).all(), "individual nodes were removed"
    assert (s_nodes[:, 0] == s_nodes[:, 1]).all(), "node mapping is incorrect"
    s_ind = sts.nodes_individual[new_nodes[:, 0]]
    # Assert that we've mapped the individuals correctly
    assert (
        ts.individuals_location[sampled["ind"]] == sts.individuals_location[s_ind]
    ).all()
    assert (ts.individuals_time[sampled["ind"]] == sts.individuals_time[s_ind]).all()
    assert (ts.individuals_flags[sampled["ind"]] == sts.individuals_flags[s_ind]).all()

    return sts, s_ind


def msprime_neutral_mutations(
    ts: tskit.TreeSequence, mu: float, seed: int
) -> tskit.TreeSequence:
    st = get_ts_sampling_times(ts, normalize=False)
    gt = np.array(
        ts.metadata["SLiM"]["user_metadata"]["generation_times"], dtype=np.float64
    )
    mean_gt = gt[st.min() : st.max()].mean()
    return msprime.sim_mutations(ts, rate=mu / mean_gt, random_seed=seed)


def simplify_and_mutate_tree_sequence(
    in_path: Path, out_path: Path, sampled: pl.DataFrame, mu: float, seed: int
) -> pl.DataFrame:
    assert out_path.parent.exists(), f"{out_path.parent} does not exist"
    ts = tszip.load(in_path)
    # obtain simplified tree sequence and our new sampled individual ids
    ts, s_ind = simplify_tree_sequence(ts, sampled)
    ts = msprime_neutral_mutations(ts, mu, seed)
    ind_nodes_simplified = np.vstack([ts.individual(i).nodes for i in s_ind])
    tszip.compress(ts, out_path)
    # TODO: store generation time here!!
    return sampled.select("run_id", "ind").with_columns(
        pl.Series("s_ind", s_ind),
        pl.Series("ind_nodes", ind_nodes_simplified, pl.Array(pl.Int32, 2)),
    )


def simplify_and_mutate_tree_sequences(
    in_paths: dict[str, Path],
    out_paths: dict[str, Path],
    out_data_path: Path,
    sample_data: pl.DataFrame,
    sample_meta_path: Path,
    mu: int,
    n_jobs: int,
    verbose: int = 10,
    **process_kwargs: Any,
) -> None:
    assert set(in_paths) == set(
        out_paths
    ), "in and out paths must have the same run_ids"
    seeds = get_seed(len(in_paths))
    assert isinstance(seeds, np.ndarray)  # mypy
    result_iter = cast(
        Generator[pl.DataFrame, None, None],
        Parallel(verbose=verbose, n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(simplify_and_mutate_tree_sequence)(
                in_paths[run_id],
                out_paths[run_id],
                sampled=sample_data.filter(pl.col("run_id") == run_id),
                mu=mu,
                seed=seed,
            )
            for run_id, seed in zip(in_paths, seeds)
        ),
    )
    ind_map = pl.DataFrame()
    # NB no need to rechunk because we're saving to disk immediately
    for data in result_iter:
        ind_map.vstack(data, in_place=True)
    out_sample_data = sample_data.join(
        ind_map, on=["run_id", "ind"], how="left", coalesce=True
    ).sort("run_id", "sampling_time", "s_ind")
    write_parquet(
        out_sample_data, out_data_path, metadata_from=sample_meta_path, **process_kwargs
    )
