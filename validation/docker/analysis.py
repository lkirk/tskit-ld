import argparse
import logging
import os
import sys

import msprime
import numpy as np

TSKIT_PROTO_DIR = "/tskit/python"
if TSKIT_PROTO_DIR not in sys.path:
    sys.path.insert(0, TSKIT_PROTO_DIR)
from tests.test_ld_matrix import ld_matrix, positions_to_tree_indices, get_index_repeats


def get_upper_tri_no_diag(arr):
    idx = np.zeros_like(arr, dtype=bool)
    idx[np.triu_indices_from(arr)] = True
    idx &= ~np.eye(len(arr), dtype=bool)
    return idx


def upper_off_diag_mean(arr):
    return arr[get_upper_tri_no_diag(arr)].mean()


def position_array(ts, stat_arr, pos):
    idx = positions_to_tree_indices(ts.breakpoints(as_array=True), pos)
    result = np.zeros((len(idx), len(idx)))
    repeats = get_index_repeats(idx)
    row = 0
    for r in range(ts.num_trees):
        col = 0
        for c in range(ts.num_trees):
            for i in range(repeats[r]):
                for j in range(repeats[c]):
                    result[i + row, j + col] = stat_arr[r, c]
            col += repeats[c]
        row += repeats[r]
    return result


def get_mean_stat_for_all_positions(ts, stat_mat, positions):
    return np.array(
        [
            upper_off_diag_mean(position_array(ts, stat_mat, positions[k]))
            for k in positions
        ]
    )


def compute_mean_stat_for_recomb_distances(ts, positions, mu):
    d2 = ld_matrix(ts, stat="d2_unbiased", mode="branch")
    pi2 = ld_matrix(ts, stat="pi2_unbiased", mode="branch")
    dz = ld_matrix(ts, stat="dz_unbiased", mode="branch")
    return np.vstack(
        [
            get_mean_stat_for_all_positions(ts, d2, positions),
            get_mean_stat_for_all_positions(ts, dz, positions),
            get_mean_stat_for_all_positions(ts, pi2, positions),
        ]
    ) * (mu**2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rep")
    parser.add_argument("seeds")
    args = parser.parse_args()

    seeds = np.fromstring(args.seeds, sep="-", dtype=np.int32)
    assert not (seeds < 1).any(), "bad seeds"

    Ne = 10_000
    mu = 2e-8
    r = 1.25e-5
    sequence_length = 11

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    positions = dict()
    for i in range(1, sequence_length - 1):
        positions[i] = np.arange(0, sequence_length, r * i * Ne)

    logger.info("Processing rep=%s seeds=%s cwd=%s", args.rep, args.seeds, os.getcwd())
    results = []
    for i, seed in enumerate(seeds):
        logger.info("Starting round %d seed %d", i, seed)
        ts = msprime.sim_ancestry(
            samples=10,
            population_size=Ne,
            recombination_rate=r,
            sequence_length=sequence_length,
            random_seed=seed,
        )
        results.append(compute_mean_stat_for_recomb_distances(ts, positions, mu))
        logger.info("Round %d complete", i)

    np.savez_compressed(args.rep, np.dstack(results).mean(2))


if __name__ == "__main__":
    main()
