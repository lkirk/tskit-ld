import demes
import moments
import moments.LD
from moments.LD.LDstats_mod import LDstats

import numpy.typing as npt
import numpy as np

from .ld_decay import midpoint


def simpson(edge: npt.NDArray, mid: npt.NDArray) -> npt.NDArray:
    return (edge[:-1] + 4 * mid + edge[1:]) / 6


def gather_moments_data_steady_state(
    rho: float,
    theta: float,
    bins: npt.NDArray,
    demog: demes.demes.Graph,
    sampling_times: list[int],
    n_pop: int,
    Ne: int,
) -> LDstats:
    # NB: only works for our very specific scenario
    mig_matrix = np.asarray(demog.migration_matrices()[0][0]) * 2 * Ne
    nus = [0.5] * n_pop
    frozen = None
    if n_pop > 1:
        # TODO: for now, make this assertion
        if sampling_times[0] > sampling_times[1]:
            raise ValueError(
                "we expect first sampling time to be <= second sampling "
                f"time, got: {sampling_times}"
            )
    ld_stats = LDstats(
        moments.LD.Numerics.steady_state(
            nus, m=mig_matrix, rho=rho * bins, theta=theta
        ),
        num_pops=n_pop,
    )
    # Also not sure if this condition is completely necessary
    if sampling_times != [0, 0]:
        if n_pop > 1:
            # NB: holding pop B constant, not general to all scenarios
            frozen = [False, True]
        if n_pop > 1:
            sampling_time = sampling_times[1] - sampling_times[0]
        else:
            sampling_time = sampling_times[0]
        ld_stats.integrate(
            nus,
            tf=sampling_time / (2 * Ne),
            rho=rho * bins,
            theta=theta,
            m=mig_matrix.copy(),  # TODO: moments mutates mig matrix
            frozen=frozen,
        )
    return ld_stats


def gather_moments_data_demog_2_pop(
    rho: float,
    theta: float,
    bins: npt.NDArray,
    sampled_demes: list[str],
    demog: demes.demes.Graph,
    sampling_times: list[int],
    from_steady_state: bool = False,
    Ne: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if from_steady_state:
        if Ne is None:
            raise ValueError("Ne must be specified if computing from steady state")
        edges_result = gather_moments_data_steady_state(
            rho, theta, bins, demog, sampling_times, 2, Ne
        )
        mids_result = gather_moments_data_steady_state(
            rho, theta, midpoint(bins), demog, sampling_times, 2, Ne
        )
    else:
        edges_result = moments.Demes.LD(
            demog,
            sampled_demes=sampled_demes,
            sample_times=sampling_times,
            rho=rho * bins,
            theta=theta,
        )
        mids_result = moments.Demes.LD(
            demog,
            sampled_demes=sampled_demes,
            sample_times=sampling_times,
            rho=rho * midpoint(bins),
            theta=theta,
        )
    mids_names = mids_result.names()[0]
    edges_names = edges_result.names()[0]

    mids_ld_stats = np.vstack(mids_result[:-1])
    mids_D2_cross = mids_ld_stats[:, mids_names.index("DD_0_1")]
    mids_pi2_1 = mids_ld_stats[:, mids_names.index("pi2_0_0_0_0")]
    mids_pi2_2 = mids_ld_stats[:, mids_names.index("pi2_1_1_1_1")]

    edges_ld_stats = np.vstack(edges_result[:-1])
    edges_D2_cross = edges_ld_stats[:, edges_names.index("DD_0_1")]
    edges_pi2_1 = edges_ld_stats[:, edges_names.index("pi2_0_0_0_0")]
    edges_pi2_2 = edges_ld_stats[:, edges_names.index("pi2_1_1_1_1")]

    D2 = simpson(edges_D2_cross, mids_D2_cross)
    pi2 = simpson(
        np.sqrt(edges_pi2_1) * np.sqrt(edges_pi2_2),
        np.sqrt(mids_pi2_1) * np.sqrt(mids_pi2_2),
    )

    return D2, pi2


def gather_moments_data_demog_1_pop(
    rho: float,
    theta: float,
    bins: npt.NDArray,
    sampled_deme: str,
    demog: demes.demes.Graph,
    sampling_time: int,
    from_steady_state: bool = False,
    Ne: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if from_steady_state:
        if Ne is None:
            raise ValueError("Ne must be specified if computing from steady state")
        edges_result = gather_moments_data_steady_state(
            rho, theta, bins, demog, [sampling_time], 1, Ne
        )
        mids_result = gather_moments_data_steady_state(
            rho, theta, midpoint(bins), demog, [sampling_time], 1, Ne
        )
    edges_result = moments.Demes.LD(
        demog,
        sampled_demes=[sampled_deme],
        sample_times=[sampling_time],
        rho=rho * bins,
        theta=theta,
    )
    mids_result = moments.Demes.LD(
        demog,
        sampled_demes=[sampled_deme],
        sample_times=[sampling_time],
        rho=rho * midpoint(bins),
        theta=theta,
    )
    mids_names = mids_result.names()[0]
    edges_names = edges_result.names()[0]

    mids_ld_stats = np.vstack(mids_result[:-1])
    mids_D2 = mids_ld_stats[:, mids_names.index("DD_0_0")]
    mids_pi2 = mids_ld_stats[:, mids_names.index("pi2_0_0_0_0")]
    edges_ld_stats = np.vstack(edges_result[:-1])
    edges_D2 = edges_ld_stats[:, edges_names.index("DD_0_0")]
    edges_pi2 = edges_ld_stats[:, edges_names.index("pi2_0_0_0_0")]

    D2 = simpson(edges_D2, mids_D2)
    pi2 = simpson(edges_pi2, mids_pi2)

    return D2, pi2


def moments_sigma_d2(
    rho: float,
    theta: float,
    bins: npt.NDArray,
    sampled_demes: list[str],
    demog: demes.demes.Graph,
    sampling_times: list[int],
    n_pop: int,
    from_steady_state: bool = False,
    Ne: int | None = None,
) -> npt.NDArray[np.float64]:
    if n_pop == 1:
        D2, pi2 = gather_moments_data_demog_1_pop(
            rho,
            theta,
            bins,
            sampled_demes[0],
            demog,
            sampling_times[0],
            from_steady_state,
            Ne,
        )
    elif n_pop == 2:
        D2, pi2 = gather_moments_data_demog_2_pop(
            rho,
            theta,
            bins,
            sampled_demes,
            demog,
            sampling_times,
            from_steady_state,
            Ne,
        )
    else:
        raise Exception
    return D2 / pi2
