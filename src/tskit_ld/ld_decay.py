import concurrent.futures
import numpy.typing as npt
from collections.abc import Iterable
from itertools import zip_longest
from typing import Any, Generator, Optional

import numpy as np
import tskit

from .types import NPInt64Array, NPFloat64Array

type DecayReturnType = tuple[NPInt64Array, NPInt64Array, NPFloat64Array]


def midpoint(bins: npt.NDArray) -> npt.NDArray:
    return (bins[1:] + bins[:-1]) / 2


def get_max_dist_slice(pos: list[int], max_dist: int) -> Generator[slice, None, None]:
    bounds = np.vstack([pos, pos + np.repeat(max_dist, len(pos))]).T
    for start, stop in np.searchsorted(pos, bounds):
        yield slice(start + 1, stop)


def chunks(iterable: Iterable, n: int) -> Generator[Any, None, None]:
    args = [iter(iterable)] * n
    i = 0
    for chunk in zip_longest(*args):
        yield i, tuple(filter(None, chunk))
        i += n


def bincount_unique(x: npt.NDArray, weights: npt.NDArray) -> npt.NDArray[np.int64]:
    # passes the following test...
    # bincount_unique(arr, np.ones_like(arr)) == np.unique(arr, return_counts=True)[1]
    # find breakpoints with closed invervals, starting with 0

    breaks = np.insert(np.where(x[1:] != x[:-1])[0] + 1, 0, 0)
    return np.add.reduceat(weights, breaks)


def ld_decay(
    ts: tskit.TreeSequence,
    chunk_size: int,
    n_threads: int,
    max_dist: int,
    win_size: Optional[int] = None,
    bins: Optional[npt.NDArray] = None,
    **ld_kwargs,
) -> DecayReturnType:
    sites = np.arange(ts.num_sites, dtype=np.int32)
    pos = ts.tables.sites.position
    if bins is None:
        bins = np.arange(0, max_dist + 1, step=win_size, dtype=np.int64)
    assert len(ld_kwargs.get("sample_sets", [])) <= 1, "only one sample set allowed"

    def worker(args):
        chunk_idx, chunk = args
        result = np.zeros(len(bins) - 1, dtype=np.float64)
        bin_count = np.zeros(len(bins) - 1, dtype=np.int64)
        chunk_slice = slice(chunk[0].start - 1, chunk[-1].stop)
        if "sample_sets" in ld_kwargs:
            ld = ts.ld_matrix(
                sites=[sites[chunk_idx : chunk_idx + chunk_size], sites[chunk_slice]],
                **ld_kwargs,
            )[0]
        else:
            ld = ts.ld_matrix(
                sites=[sites[chunk_idx : chunk_idx + chunk_size], sites[chunk_slice]],
                **ld_kwargs,
            )
        for k, (j, s) in enumerate(enumerate(chunk, chunk_idx)):
            ld_row = ld[k, sites[s] - chunk_idx]  # implicit copy
            bin_idx = np.searchsorted(bins[1:], pos[s] - pos[j])
            bin_idx = bin_idx[~np.isnan(ld_row)]
            ld_row = ld_row[~np.isnan(ld_row)]
            if len(ld_row) == 0:
                continue
            bin_idx_uniq, bc = np.unique(bin_idx, return_counts=True)
            bin_count[bin_idx_uniq] += bc
            result[bin_idx_uniq] += bincount_unique(bin_idx, ld_row)
        return result, bin_count

    pool = None
    try:
        work = chunks(get_max_dist_slice(pos, max_dist), chunk_size)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as pool:
            results = pool.map(worker, work)
    except KeyboardInterrupt as e:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        raise e
    result = []
    bin_count = []
    for r, bc in results:
        result.append(r)
        bin_count.append(bc)
    count = np.vstack(bin_count).sum(0)
    return bins, count, np.vstack(result).sum(0) / count


def ld_decay_two_way(
    ts: tskit.TreeSequence,
    chunk_size: int,
    n_threads: int,
    max_dist: int,
    win_size: Optional[int] = None,
    bins: Optional[npt.NDArray] = None,
    **ld_kwargs,
) -> DecayReturnType:
    sites = np.arange(ts.num_sites, dtype=np.int32)
    pos = ts.tables.sites.position
    if bins is None:
        bins = np.arange(0, max_dist + 1, step=win_size, dtype=np.int64)

    def worker(args):
        chunk_idx, chunk = args
        result = np.zeros(len(bins) - 1, dtype=np.float64)
        bin_count = np.zeros(len(bins) - 1, dtype=np.int64)
        chunk_slice = slice(chunk[0].start - 1, chunk[-1].stop)
        ld = ts.ld_matrix_two_way(
            sites=[sites[chunk_idx : chunk_idx + chunk_size], sites[chunk_slice]],
            **ld_kwargs,
        )
        for k, (j, s) in enumerate(enumerate(chunk, chunk_idx)):
            ld_row = ld[k, sites[s] - chunk_idx]  # implicit copy
            bin_idx = np.searchsorted(bins[1:], pos[s] - pos[j])
            bin_idx = bin_idx[~np.isnan(ld_row)]
            ld_row = ld_row[~np.isnan(ld_row)]
            if len(ld_row) == 0:
                continue
            bin_idx_uniq, bc = np.unique(bin_idx, return_counts=True)
            bin_count[bin_idx_uniq] += bc
            result[bin_idx_uniq] += bincount_unique(bin_idx, ld_row)
        return result, bin_count

    pool = None
    try:
        work = chunks(get_max_dist_slice(pos, max_dist), chunk_size)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as pool:
            results = pool.map(worker, work)
    except KeyboardInterrupt as e:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        raise e
    result = []
    bin_count = []
    for r, bc in results:
        result.append(r)
        bin_count.append(bc)
    count = np.vstack(bin_count).sum(0)
    return bins, count, np.vstack(result).sum(0) / count
