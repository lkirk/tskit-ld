import matplotlib.pyplot as plt
import msprime
import numpy as np
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
    num_sites = np.array([t.num_sites for t in ts.trees()])
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
    return np.stack([out, count])


def zero_diag(a):
    """
    Zero out the diagonal of a given matrix.
    Will throw an error if matrix is not square.
    Use the return value or not, a copy is not made.
    """
    a[np.diag_indices_from(a)] = 0
    return a


def ts_exec(ts, attr, *args, **kwargs):
    return getattr(ts, attr)(*args, **kwargs)


def ts_exec_and_sum(ts, attr, *args, **kwargs):
    # if kwargs.pop("nodiag", None) is True:
    return tree_sum(ts, zero_diag(ts_exec(ts, attr, *args, **kwargs)))
    # return tree_sum(ts, ts_exec(ts, attr, *args, **kwargs))


def parallel(it, **kwargs):
    return Parallel(
        # defaults if parameter not specified
        n_jobs=kwargs.pop("n_jobs", 15),
        verbose=kwargs.pop("verbose", 1),
        return_as=kwargs.pop("return_as", "list"),
    )(it)


def site_ld_matrix(ts, seed, n_reps, mu, *ldargs, pkw=None, **ldkw):
    """
    Data Dimensions: [rep, stat=0/count=1, siterow, sitecol]
    """
    pkwargs = dict() if pkw is None else pkw
    return np.stack(
        parallel(
            (
                delayed(ts_exec_and_sum)(mts, "ld_matrix", *ldargs, mode="site", **ldkw)
                for mts in gen_mut_sims(
                    ts, seed, n_reps, rate=mu, discrete_genome=False
                )
            ),
            **pkwargs,
        )  # type: ignore
    )  # type: ignore


def compute_ld(ts, stats, seed, n_reps, mu, verbose=1):
    branch_mats = {
        s: mat
        for s, mat in zip(
            stats,
            parallel(
                (
                    delayed(ts_exec)(ts, "ld_matrix", mode="branch", stat=s)
                    for s in stats
                ),
                verbose=verbose,
            ),
        )
    }
    return {
        s: (
            site_ld_matrix(
                ts,
                seed,
                n_reps=n_reps,
                mu=mu,
                stat=s,
                pkw=dict(verbose=verbose),
            ),
            branch_mats[s],
        )
        for s in stats
    }


def rel_err(obs, exp):
    return (obs - exp) / np.abs(exp)


def compare(site, branch, n_reps, L2, mu):
    s = site[:, 0, :, :].sum(0) / L2 / n_reps
    b = branch * mu**2
    return s, b, rel_err(s, b) * 100


def print_compare(site, branch, n_reps, L2, mu):
    s, b, err = compare(site, branch, n_reps, L2, mu)
    print("site\n", s, "\nbranch\n", b, "\nRelative Percent Error\n", err)


def plot_compare(stats, n_reps, L2, mu):
    assert len(stats) <= 10
    fig, axes = plt.subplots(
        2,
        5,
        figsize=(2 * 5, 2 * 2),
        subplot_kw=dict(box_aspect=1),
        layout="constrained",
    )
    for ax, (stat, (site, branch)) in zip(axes.flatten(), stats.items()):
        s, b = [m.flatten() for m in compare(site, branch, n_reps, L2, mu)[:-1]]
        xy_line = np.linspace(np.min([s, b]), np.max([s, b]))
        ax.scatter(s, b, s=5)
        ax.plot(xy_line, xy_line, c="C1", alpha=0.8)
        ax.set_title(STAT_TEX[stat])
    fig.supylabel("Branch stat")
    fig.supxlabel("Site stat")
