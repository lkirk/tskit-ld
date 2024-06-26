import json
from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Callable, cast, Generator, Optional

import matplotlib.pyplot as plt

# import msprime
import numpy as np
import numpy.typing as npt
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pyslim
import tskit
import tszip
from joblib import Parallel, delayed


def get_inds_sampling_time(ts: tskit.TreeSequence) -> npt.NDArray[np.int64]:
    """Get the time at which individuals are alive"""
    # verify about time units
    assert (
        ts.metadata["SLiM"]["model_type"] == "nonWF"
        and ts.metadata["SLiM"]["stage"] == "late"
    )
    age_offset = 1  # see above assertion
    # sampling time
    sampling_times = np.array(
        ts.metadata["SLiM"]["user_metadata"]["sampling_times"], dtype=np.int64
    )
    st = pyslim.slim_time(ts, sampling_times)
    # birth time
    bt = np.repeat(ts.individuals_time, len(st)).reshape(-1, len(st))
    # sample ages at sampling
    age = np.repeat(ts.tables.individuals.metadata_vector("age"), len(st)).reshape(
        -1, len(st)
    )
    return st[np.where((bt >= st) & (bt - age < st + age_offset))[1]]


def preprocess_metadata(meta: dict[str, Any]) -> bytes:
    """Strip out unnecessary metadata, flatten and convert values to bytes"""
    assert len(meta["SLiM"]["user_metadata"]["params"]) == 1
    assert all(
        [len(v) == 1 for v in meta["SLiM"]["user_metadata"]["params"][0].values()]
    )
    meta["params"] = {
        k: v[0] for k, v in meta["SLiM"]["user_metadata"]["params"][0].items()
    }
    del meta["SLiM"]["user_metadata"]
    meta = {**meta["SLiM"], **meta["params"]}  # flatten dict a bit
    return json.dumps(meta).encode("utf-8")


def load_ts_and_process_spatial_data(
    ts_path: Path,
    run_id: str,
    run_ids: pl.Enum,
) -> tuple[pl.DataFrame, dict[str, bytes]]:
    ts = cast(tskit.TreeSequence, tszip.load(ts_path))
    assert (
        np.array([t.num_roots for t in ts.trees()]) == 1
    ).all(), "not all trees have coalesced"
    sampling_time = get_inds_sampling_time(ts)
    assert (ts.individuals_location[:, 2] == 0).all()
    df = pl.DataFrame(
        {
            "ind": np.arange(ts.num_individuals, dtype=np.uint64),
            "x": ts.individuals_location[:, 0],
            "y": ts.individuals_location[:, 1],
            "sampling_time": sampling_time,
            "age": ts.tables.individuals.metadata_vector("age"),
        }
    ).with_columns(run_id=pl.lit(run_id, dtype=run_ids))
    return df, {run_id: preprocess_metadata(ts.metadata)}


def process_data_and_write_parquet(
    in_paths: dict[str, Path],
    out_path: Path,
    n_jobs: int,
    verbose: int = 10,
    **process_kwargs: Any,
) -> None:
    assert out_path.parent.exists(), f"{out_path} does not exist"
    run_ids = pl.Enum(in_paths.keys())
    result_iter = cast(
        Generator[tuple[pl.DataFrame, dict[str, bytes]], None, None],
        Parallel(verbose=verbose, n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(load_ts_and_process_spatial_data)(f, run_id, run_ids)
            for run_id, f in in_paths.items()
        ),
    )
    df = pl.DataFrame()
    meta = dict()
    # NB no need to rechunk because we're saving to disk immediately
    for data, m in result_iter:
        df.vstack(data, in_place=True)
        meta.update(m)
    write_parquet(df, out_path, meta, **process_kwargs)


def write_parquet(
    data: pl.DataFrame | pa.Table,
    out_path: Path,
    metadata,
    compression: str = "ZSTD",
    compression_level: Optional[int] = None,
) -> None:
    match data:
        case pl.DataFrame():
            table = data.to_arrow()
        case pa.Table():
            table = data
        case _:
            raise ValueError

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


def get_metadata_df(in_path: Path) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {**dict(run_id=k.decode("utf-8")), **json.loads(v.decode("utf-8"))}
            for k, v in pq.read_metadata(in_path).metadata.items()
            if k != b"ARROW:schema"
        ]
    )


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


def get_individuals_at_time_space_rings(
    ts, t, space_width, n_rings, sample_seed=None, n_samples=None
):
    alive = pyslim.individuals_alive_at(ts, t)
    n_alive = len(alive)
    locs = ts.individuals_location[alive, 0:2]
    bounds = np.linspace(0, space_width / 2, n_rings)
    lower_bounds = np.tile(bounds[:-1], n_alive).reshape(n_alive, len(bounds) - 1)
    upper_bounds = np.tile(bounds[1:], n_alive).reshape(n_alive, len(bounds) - 1)
    center = np.repeat(space_width / 2, np.prod(locs.shape)).reshape(*locs.shape)
    dist = np.linalg.norm(center - locs, axis=1)[:, np.newaxis]
    in_range = (lower_bounds <= dist) & (dist < upper_bounds)
    assert (in_range.sum(1) <= 1).all()
    if sample_seed is not None:
        assert n_samples is not None
        rng = np.random.RandomState(sample_seed)
        return {
            str(b): rng.choice(alive[in_range[:, i]], size=n_samples, replace=False)
            for i, b in enumerate(bounds[1:])
        }
    return {
        "all": alive,
        **{str(b): alive[in_range[:, i]] for i, b in enumerate(bounds[1:])},
    }


from dataclasses import dataclass


@dataclass
class PairwiseStats:
    pair_func: Callable
    nodes: np.array
    samples: np.array
    pairs: np.array
    geog_distance: np.array
    divergence: np.array
    groups: list[str]
    ind_groups: np.array

    def __init__(
        self,
        ts: tskit.TreeSequence,
        grouped: dict[str, np.array],
        pair_func=lambda l: list(combinations_with_replacement(l, 2)),
    ):
        self.pair_func = pair_func
        self.nodes = np.vstack(
            [ts.individual(i).nodes for v in grouped.values() for i in v]
        )
        self.samples = np.hstack(list(grouped.values()))
        self.pairs = np.vstack(pair_func(range(len(self.samples))))
        self.geog_distance = get_geog_distance(ts, self.samples[self.pairs])
        self.divergence = ts.divergence(self.nodes, indexes=self.pairs)
        self.groups = list(grouped)
        self.ind_groups = np.hstack([[k] * len(v) for k, v in grouped.items()])


def get_geog_distance(ts, individual_pairs):
    a, b = ts.individuals_location[individual_pairs, 0:2].swapaxes(0, 1)
    return np.linalg.norm(a - b, axis=1)


def compute_stats_parallel(ts, groups, n_jobs=15):
    return Parallel(verbose=10, n_jobs=n_jobs)(
        delayed(PairwiseStats)(ts, s) for s in groups
    )
