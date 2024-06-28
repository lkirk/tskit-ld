import json
from itertools import combinations_with_replacement
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Optional,
    Sequence,
    SupportsIndex,
    cast,
)

import matplotlib.pyplot as plt
import msprime
import numpy as np
import numpy.typing as npt
import polars as pl
import polars.type_aliases
import pyarrow as pa
import pyarrow.parquet as pq
import pyslim
import tskit
import tszip
from joblib import Parallel, delayed
from more_itertools import one

SEED = 23
RNG = np.random.RandomState(23)

# Type alias for numpy
NPShapeLike = Optional[Sequence | SupportsIndex]  # (as defined in the _typing module)
NPInt64Array = npt.NDArray[np.int64]
NPUInt32Array = npt.NDArray[np.uint32]
NPInt32Array = npt.NDArray[np.int32]

# Type alias for polars
IntoExpr = polars.type_aliases.IntoExpr


def get_seed(size: NPShapeLike = None) -> int | NPInt64Array:
    """
    Generate random seeds for sampling processes. We seed our RNG to
    obtain some determinism for the pipeline, but generate random seeds
    for each step.

    NB: if the ordering changes between steps, seeds (and results) will change
    """
    return RNG.randint(0, 2**31, size=size)


## General Data IO functionality, used in most steps


def load_ts(ts_path: Path) -> tskit.TreeSequence:
    return cast(tskit.TreeSequence, tszip.load(ts_path))


def write_parquet(
    data: pl.DataFrame | pa.Table,
    out_path: Path,
    metadata: Optional[dict[str, bytes] | dict[bytes, bytes]] = None,
    metadata_from: Optional[Path] = None,
    compression: str = "ZSTD",
    compression_level: Optional[int] = None,
) -> None:
    """
    Write parquet file from either a polars dataframe or a pyarrow table.
    Include user specified metadata in the form of a flat dictionary.
    We use ZSTD compression with the default compression level.
    """
    match data:
        case pl.DataFrame():
            table = data.to_arrow()
        case pa.Table():
            table = data
        case _:
            raise ValueError

    if metadata is None and metadata_from is None:
        raise ValueError("One of `metadata` or `metadata_from` is required")
    elif metadata is not None and metadata_from is not None:
        raise ValueError("`metadata` and `metadata_from` are mutually exclusive")

    if metadata_from is not None:
        metadata = cast(dict[bytes, bytes], pq.read_metadata(metadata_from).metadata)
        del metadata[b"ARROW:schema"]

    # TODO: writing with pyarrow turns our enum type into a categorical
    #       either figure out how to preserve this or remove the complexity
    #       of creating Enum types.
    #       hints here: https://github.com/pola-rs/polars/pull/13943
    with pq.ParquetWriter(
        out_path,
        table.schema.with_metadata(metadata),
        compression=compression,
        compression_level=compression_level,
    ) as writer:
        writer.write_table(table)


def read_metadata_df(in_path: Path) -> pl.DataFrame:
    file_meta = pq.read_metadata(in_path).metadata
    del file_meta[b"ARROW:schema"]
    file_meta = sorted(
        file_meta.items(),
        key=lambda k: tuple(map(int, k[0].decode("utf-8").split("-"))),
    )
    meta = pl.DataFrame(
        [
            {**dict(run_id=k.decode("utf-8")), **json.loads(v.decode("utf-8"))}
            for k, v in file_meta
        ]
    )
    run_id_enum = pl.Enum(meta["run_id"])
    meta = meta.with_columns(run_id=pl.col("run_id").cast(run_id_enum))
    return meta


def read_parquet_file(
    in_path: Path, collect: bool = False
) -> tuple[pl.DataFrame, pl.LazyFrame | pl.DataFrame]:
    """
    Read parquet file and associated metadata. We have to work around the
    fact that serialization into parquet format converts our run_id to a
    categorical variable. Once this is fixed, we won't have to perform our
    cast to the enum type.
    """
    meta = read_metadata_df(in_path)
    data = pl.scan_parquet(in_path).with_columns(
        run_id=pl.col("run_id").cast(meta["run_id"].dtype)
    )
    if collect:
        return meta, data.collect()
    return meta, data


def merge_parquet_files(
    in_paths: Iterable[Path],
    out_path: Path,
    n_jobs: int,
    metadata: Optional[dict[str, bytes] | dict[bytes, bytes]] = None,
    metadata_from: Optional[Path] = None,
    verbose: int = 10,
) -> None:
    if out_path.exists():
        raise ValueError(
            f"{out_path} exists, performing this operation will append to the existing file, remove to continue."
        )
    if metadata is None and metadata_from is None:
        raise ValueError("One of `metadata` or `metadata_from` is required")
    elif metadata is not None and metadata_from is not None:
        raise ValueError("`metadata` and `metadata_from` are mutually exclusive")

    if metadata_from is not None:
        metadata = cast(dict[bytes, bytes], pq.read_metadata(metadata_from).metadata)
        del metadata[b"ARROW:schema"]

    # Merge files directly to disk in parallel
    Parallel(verbose=verbose, n_jobs=n_jobs)(
        delayed(lambda p: pl.scan_parquet(p).sink_parquet(out_path))(p)
        for p in in_paths
    )

    schema = pq.read_schema(out_path)
    pq.write_metadata(schema.with_metadata(metadata), out_path)


## Initial Raw Sample Data Processing


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


def preprocess_metadata(meta: dict[str, Any]) -> bytes:
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
    return json.dumps(meta).encode("utf-8")


def read_ts_and_process_spatial_data(
    ts_path: Path,
    run_id: str,
    run_ids: pl.Enum,
) -> tuple[dict[str, bytes], pl.DataFrame]:
    ts = load_ts(ts_path)
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


def linspace(start, stop, num, endpoint=True):
    delta = stop - start
    div = (num - 1) if endpoint else num
    step = delta / div
    y = (pl.arange(0, num) * step) + start
    if endpoint:
        # lazy equivalent to y[-1] = stop
        y = y.shift(1).shift(-1, fill_value=stop)
    return y


def dist_from_point(point):
    x2 = ((pl.col("x") * -1) + point) ** 2
    y2 = ((pl.col("y") * -1) + point) ** 2
    return (x2 + y2).sqrt()


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
        # run.filter(pl.col("sample_group").is_not_null())
        df.with_columns(sample_group=sample_group_fn)
        .filter(pl.col("sample_group").is_not_null())
        .group_by("sampling_time", "run_id", "sample_group")
        .agg(ind=pl.col("ind").sample(n_ind, seed=cast(int, get_seed())))
        .explode("ind")
    )


def spatially_sample_individuals_join_data(
    in_path: Path, n_ind: int, sample_group_fn: IntoExpr
) -> pl.DataFrame:
    _, df = read_parquet_file(in_path)
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
        ts.simplify(
            samples=ind_nodes.reshape(-1), keep_input_roots=True, map_nodes=True
        ),
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
    assert (new_nodes.flatten() == sts.samples()).all()

    return sts, s_ind


def simplify_tree_sequence_noinputroots(
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
    ts = load_ts(in_path)
    # obtain simplified tree sequence and our new sampled individual ids
    ts, s_ind = simplify_tree_sequence(ts, sampled)
    ts = msprime_neutral_mutations(ts, mu, seed)
    tszip.compress(ts, out_path)
    return sampled.select("run_id", "ind").with_columns(pl.Series("s_ind", s_ind))


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


## Summary statistics


def compute_divergence_and_geog_distance(
    ts: tskit.TreeSequence, pairs: NPInt64Array, ind: pl.Series
) -> pl.Series:
    ind_nodes = np.vstack([ts.individual(i).nodes for i in ind])
    a, b = ts.individuals_location[pairs, 0:2].swapaxes(0, 1)
    dist = np.linalg.norm(a - b, axis=1)
    div = ts.divergence(ind_nodes, indexes=pairs)
    ind_pairs = ind.to_numpy()[pairs]
    return (
        pl.DataFrame({"geog_dist": dist, "divergence": div, "ind_pairs": ind_pairs})
        .to_struct()
        .alias("stats")
    )


def compute_divergence_and_geog_distance_for_sim(
    ts: tskit.TreeSequence,
    df: pl.DataFrame,
    pair_func: Callable[..., list[tuple[int, int]]] = lambda l: list(
        combinations_with_replacement(l, 2)
    ),
) -> pl.DataFrame:
    n_inds_samp_time = one(
        df.group_by("sampling_time").agg(pl.col("s_ind").count())["s_ind"].unique(),
        too_short=ValueError("group lengths differ"),
    )
    pairs = np.array(list(pair_func(range(n_inds_samp_time))), dtype=np.int64)
    return (
        df.group_by("sampling_time")
        .agg(
            pl.col("s_ind").map_batches(
                lambda g: compute_divergence_and_geog_distance(ts, pairs, g),
                return_dtype=pl.Struct,
            )
        )
        .with_columns(
            run_id=pl.lit(one(df["run_id"].unique()), dtype=df["run_id"].dtype)
        )
    )


## Plotting functionality


def plot_sample_locations(
    df, space_width=35, space_height=15, plot_scale=3.5, ax=None, sampled_s=20, all_s=7
):
    set_labels = False
    if ax is None:
        _, ax = plt.subplots(
            figsize=(space_width / plot_scale, space_height / plot_scale)
        )
        set_labels = True
    parts = (
        df.sort("sample_group")
        .with_columns(pl.col("sample_group") + 1)
        .fill_null(0)
        .partition_by(["sample_group"], include_key=False, as_dict=True)
    )
    keys = list(parts)
    k = keys[0]
    p = parts[k]
    ax.scatter(
        p.select(pl.col("x")),
        p.select(pl.col("y")),
        s=all_s,
        alpha=0.6,
        label="unsampled",
    )
    for k in keys[1:]:
        p = parts[k]
        ax.scatter(
            p.select(pl.col("x")),
            p.select(pl.col("y")),
            s=sampled_s,
            label=f"group {one(k)}",
        )
    if set_labels:
        plt.title("Sample group locations in geographic space")
        plt.ylabel("Y position")
        plt.xlabel("X position")
        plt.legend()


def plot_divergence(div, dist, indexes, groups, pairs, ax=None):
    colors = np.zeros(len(indexes))
    assert len(div) == len(dist)
    mask = np.zeros_like(div, dtype=bool)
    for i, p in enumerate(pairs):
        assert (groups == p).all(1).sum() > 0, f"{p}"
        colors[(groups == p).all(1)] = i
        mask[(groups == p).all(1)] |= True
    if ax is None:
        _, ax = plt.subplots()
    scatter = ax.scatter(dist[mask], div[mask] * 1e3, c=colors[mask], alpha=0.3)
    if ax is None:
        ax.set_xlabel("geographic distance")
        ax.set_ylabel("genetic distance (diffs/Kb)")
        ax.legend(scatter.legend_elements()[0], pairs)
    else:
        return scatter.legend_elements()[0]


def plot_divergence_upper_tri(div, dist, ind_groups, groups):
    assert len(div) == len(dist)
    n_groups = len(groups)
    fig, axes = plt.subplots(
        n_groups,
        n_groups,
        figsize=(n_groups * 3, n_groups * 3),
        sharex=True,
        sharey=True,
    )
    for (i, g1), (j, g2) in combinations_with_replacement(enumerate(groups), 2):
        p = (g1, g2)
        axes[i, j].scatter(
            dist[(ind_groups == p).all(1)] * 1e3,
            div[(ind_groups == p).all(1)],
            alpha=0.3,
            s=4,
        )
        axes[i, j].set_title((g1, g2))
        if i != j:
            axes[j, i].scatter(
                dist[(ind_groups == p).all(1)] * 1e3,
                div[(ind_groups == p).all(1)],
                alpha=0.3,
                s=4,
            )
            axes[j, i].set_title((g2, g1))
    fig.supylabel("Genetic distance (diffs/Kb)")
    fig.supxlabel("Geographic distance")
    fig.tight_layout()
