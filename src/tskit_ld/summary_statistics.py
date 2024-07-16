from itertools import combinations_with_replacement
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import tskit
from more_itertools import one

from .io import read_parquet_file
from .types import NPInt64Array


def prepare_divergence_geog_dist_cluster_params(in_path: Path, out_dir: Path) -> None:
    df = read_parquet_file(in_path)
    assert isinstance(df, pl.LazyFrame)  # mypy
    partitions = (
        df.drop("sample_group", "ind", "x", "y", "age")
        .sort("run_id", "sampling_time", "s_ind")
        .collect()
        .partition_by(["run_id"], as_dict=True, maintain_order=True)
    )
    out_dir.mkdir()
    for (run_id,), d in partitions.items():
        d.write_parquet(out_dir / f"{run_id}.parquet")


def compute_divergence_and_geog_distance(
    ts: tskit.TreeSequence, pairs: NPInt64Array, ind: pl.Series
) -> pl.Series:
    ind_nodes = np.vstack([ts.individual(i).nodes for i in ind])
    ind_pairs = ind.to_numpy()[pairs]
    a, b = ts.individuals_location[ind_pairs, 0:2].swapaxes(0, 1)
    dist = np.linalg.norm(a - b, axis=1)
    div = ts.divergence(ind_nodes, indexes=pairs)
    return pl.DataFrame(
        {"geog_dist": dist, "divergence": div, "ind_pairs": ind_pairs}
    ).to_struct()


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
            stats=pl.col("s_ind").map_batches(
                lambda g: compute_divergence_and_geog_distance(ts, pairs, g),
                return_dtype=pl.Struct,
            )
        )
        .with_columns(
            run_id=pl.lit(one(df["run_id"].unique()), dtype=df["run_id"].dtype)
        )
    )
