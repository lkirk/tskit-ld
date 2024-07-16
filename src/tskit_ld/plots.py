from itertools import combinations_with_replacement

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from more_itertools import one


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
