from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path

import matplotlib.pyplot as plt
import msprime
import numpy as np
import pyslim
import tskit
from joblib import Parallel, delayed


# def get_breakpoints(arr1, arr2=None):
#     breaks = np.zeros(len(arr1) - 1, dtype=bool)
#     if arr2 is not None:
#         assert len(arr1) == len(arr2)
#         breaks[arr1[1:] != arr2[:-1]] = True
#     else:
#         breaks[arr1[1:] != arr1[:-1]] = True
#     breaks = np.insert(breaks, 0, True)
#     breaks = np.vstack(
#         [np.where(breaks)[0], np.append(np.where(breaks)[0][1:], len(arr1))]
#     ).T
#     return [slice(*r) for r in breaks]


@dataclass
class TreeSequenceData:
    label: str
    ts: tskit.TreeSequence
    gt: float
    model_params: dict
    gentimes: np.array
    sampling_times: np.array


def load_slim_ts_and_mutate(
    ts_path,
    mu,
    seed,
    label_func=lambda p: "_".join(
        map(str, [p.parent.name, Path(p.name).with_suffix("")])
    ),
):
    label = label_func(ts_path)
    print(f"loading {label}: {ts_path}")
    ts = tskit.load(ts_path)
    gentimes = np.array(
        ts.metadata["SLiM"]["user_metadata"]["generation_times"], dtype=np.float64
    )
    gentimes = gentimes[~np.isnan(gentimes)]
    assert len(gentimes) == ts.metadata["SLiM"]["tick"]

    sampling_times = np.array(
        ts.metadata["SLiM"]["user_metadata"]["sampling_times"], dtype=int
    )

    gt = gentimes[-sampling_times[0] :].mean()
    print(
        f"Estimated generation time: {gt} from first {len(gentimes[-sampling_times[0]:])} ticks"
    )
    # relative sampling times
    sampling_times -= sampling_times[0]

    model_params = {
        k: v[0] for k, v in ts.metadata["SLiM"]["user_metadata"]["params"][0].items()
    }

    if all_coalesced := (np.array([t.num_roots for t in ts.trees()]) == 1).all():
        print("all trees have coalesced")
    assert all_coalesced

    ts = msprime.sim_mutations(ts, rate=mu / gt, random_seed=seed)

    return TreeSequenceData(
        label=label,
        ts=ts,
        gt=gt,
        model_params=model_params,
        gentimes=gentimes,
        sampling_times=sampling_times,
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
        fig, ax = plt.subplots()
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
from typing import Callable


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
