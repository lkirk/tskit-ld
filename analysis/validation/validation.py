from dataclasses import dataclass, fields
from itertools import zip_longest

import matplotlib.pyplot as plt
import msprime
import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed

STAT_TEX = {
    "r2": "$r^2$",
    "r": "$r$",
    "D": "$D$",
    "D2": "$D^2$",
    "Dz": "$D_z$",
    "pi2": r"$\pi_2$",
    "D_prime": r"$D^\prime$",
    "D2_unbiased": r"$D^2$ unbiased",
    "Dz_unbiased": r"$D_z$ unbiased",
    "pi2_unbiased": r"$\pi_2$ unbiased",
}


def gen_mut_sims(ts, seed, n, *args, **kwargs):
    rng = np.random.RandomState(seed)
    seeds = rng.randint(0, 2**32, n, np.uint32)
    for seed in seeds:
        mts = msprime.sim_mutations(ts, *args, **kwargs, random_seed=seed)
        yield mts


def tree_sum(ts, data):
    num_sites = np.array([t.num_sites for t in ts.trees()], dtype=np.int64)
    out_idx = np.where(num_sites)[0]
    add_idx = np.cumsum(num_sites[out_idx[:-1]])
    count = np.outer(num_sites, num_sites)

    out = np.zeros((ts.num_trees, ts.num_trees), dtype=np.float64)

    if len(add_idx) > 0:
        if add_idx[0] != 0:
            add_idx = np.insert(add_idx, 0, 0)
        out[np.ix_(out_idx, out_idx)] = np.add.reduceat(
            np.add.reduceat(data, add_idx, axis=0), add_idx, axis=1
        )
    return out, count


def zero_diag(a):
    """
    Zero out the diagonal of a given matrix.
    Will throw an error if matrix is not square.
    Use the return value or not, a copy is not made.
    """
    a[np.diag_indices_from(a)] = 0
    return a


def ts_exec(ts, attr, *args, **kwargs):
    """
    Execute tree sequence method with arguments
    """
    return getattr(ts, attr)(*args, **kwargs)


def ts_exec_and_sum(ts, attr, *args, **kwargs):
    return tree_sum(ts, zero_diag(ts_exec(ts, attr, *args, **kwargs)))


def parallel(it, **kwargs):
    return Parallel(
        # defaults if parameter not specified
        n_jobs=kwargs.pop("n_jobs", 15),
        verbose=kwargs.pop("verbose", 1),
        return_as=kwargs.pop("return_as", "list"),
        **kwargs,
    )(it)


def site_ld_matrix(ts, seed, n_reps, mu, *ldargs, pkw=None, **ldkw):
    """
    Data Dimensions: [rep, stat=0/count=1, siterow, sitecol]
    """
    pkwargs = dict() if pkw is None else pkw
    stat = np.zeros((ts.num_trees, ts.num_trees), dtype=np.float64)
    count = np.zeros((ts.num_trees, ts.num_trees), dtype=np.int64)
    site_jobs = (
        delayed(ts_exec_and_sum)(mts, "ld_matrix", *ldargs, mode="site", **ldkw)
        for mts in gen_mut_sims(ts, seed, n_reps, rate=mu, discrete_genome=False)
    )

    pkwargs["return_as"] = "generator"
    for s, c in parallel(site_jobs, **pkwargs):  # type: ignore
        stat += s
        count += c
    return stat, count


@dataclass
class LDResult:
    branch: dict[str, npt.NDArray[np.float64]]
    site: dict[str, npt.NDArray[np.float64]]
    site_count: dict[str, npt.NDArray[np.int64]]
    stats: list[str]

    def __repr__(self):
        stats = ", ".join(self.stats)
        field_names = f"{', '.join([f.name for f in fields(self)])}"
        return f"{self.__class__.__name__}(fields=[{field_names}], stats=[{stats}])"


def compute_ld(ts, stats, seed, n_reps, mu, verbose=1):
    branch_jobs = (
        delayed(ts_exec)(ts, "ld_matrix", mode="branch", stat=s) for s in stats
    )
    branch = dict(zip(stats, parallel(branch_jobs, verbose=verbose)))
    site_jobs = (
        site_ld_matrix(ts, seed, n_reps, mu, stat=s, pkw=dict(verbose=verbose))
        for s in stats
    )
    site, count = tuple(zip(*site_jobs))
    return LDResult(
        branch=branch,  # type: ignore
        site=dict(zip(stats, site)),
        site_count=dict(zip(stats, count)),
        stats=stats,
    )


def rel_err(obs, exp):
    return (obs - exp) / np.abs(exp)


def compare(site, branch, n_reps, L2, mu):
    s = site / L2 / n_reps
    b = branch * mu**2
    return s, b, rel_err(s, b) * 100


def print_compare(site, branch, n_reps, L2, mu):
    s, b, err = compare(site, branch, n_reps, L2, mu)
    print("site\n", s, "\nbranch\n", b, "\nRelative Percent Error\n", err)


def plot_compare(stats, n_reps, L2, mu, n_rows=2):
    n_cols, add1 = divmod(len(stats.stats), n_rows)
    n_cols += 1 if add1 else 0
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2 * n_cols, 2 * n_rows),
        subplot_kw=dict(box_aspect=1),
        layout="constrained",
    )
    for ax, stat in zip_longest(axes.flatten(), stats.stats, fillvalue=None):
        if stat is None:
            ax.remove()  # type: ignore
            continue
        s, b = [
            m.flatten()
            for m in compare(stats.site[stat], stats.branch[stat], n_reps, L2, mu)[:-1]
        ]
        xy_line = np.linspace(np.min([s, b]), np.max([s, b]))
        ax.scatter(s, b, s=5)  # type: ignore
        ax.plot(xy_line, xy_line, c="C1", alpha=0.8)  # type: ignore
        ax.set_title(STAT_TEX[stat])  # type: ignore
    fig.supxlabel("Site stat")
    fig.supylabel("Branch stat")
