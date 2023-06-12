from concurrent.futures import ThreadPoolExecutor as TPexe
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
import bottleneck as bn


"""
Module with functions for scoring coverage arrays and calculating benefit of sequences
"""


def init_scoring_vec(lowcov: float) -> NDArray:
    """
    Initialize scoring vector based on target coverage.

    :param lowcov: The target coverage value.
    :return: The scoring vector.
    """
    x = np.arange(101)
    # a = lowcov * 5
    # score_vec = -gamma.cdf(x, a=a, scale=0.2) + 1
    score_vec = 1 / (np.exp(x - lowcov) + 1)
    return score_vec



def score_array(score_vec: NDArray, cov_arr: NDArray, node_size: int) -> NDArray:
    """
    Calculate scores based on the scoring vector and coverage array.

    :param score_vec: scoring vector.
    :param cov_arr: coverage array.
    :param node_size: node size.
    :return: The calculated scores.
    """
    # grab scores using multi-indexing
    carr = cov_arr // node_size  # apply resolution reduction
    carr_int = carr.astype("int")
    scores = score_vec[carr_int]
    return scores




def calc_fragment_benefit(scores: NDArray, mu: int, node_size: int, approx_ccl: NDArray, e1: bool, e2: bool) -> Tuple[NDArray, float]:
    """
    Calculate the benefit of a fragment based on scores, mu, node_size, approx_ccl, e1, and e2.

    :param scores: Fragment's position-wise scores.
    :param mu: Length of anchor bases.
    :param node_size: node size.
    :param approx_ccl: Approx of read length distribution.
    :param e1: Left-end marker.
    :param e2: Right-end marker.
    :return: The calculated benefit and smu_sum.
    """
    # expand score to account for contig ends
    mu_ds = mu // node_size
    ccl_ds = approx_ccl // node_size
    ccl_max = ccl_ds[-1]
    sx = _expand_scores(scores, e1, e2, ccl_max)
    smu = _calc_smu_moving(score=sx, mu_ds=mu_ds)
    benefit = _calc_benefit_moving(score=sx, ccl_ds=ccl_ds)
    smu_sum = float(np.sum(smu))
    b = benefit - smu
    b[b < 0] = 0
    b = b[:, ccl_max: -ccl_max]
    assert b.shape[1] == scores.shape[0]
    return b, smu_sum



def _expand_scores(scores: NDArray, e1: bool, e2: bool, ccl_max: int) -> NDArray:
    """
    Expand scores to account for contig ends.

    :param scores: Fragment scores.
   :param e1: Left-end marker.
    :param e2: Right-end marker.
    :param ccl_max: Max of approx read length dist.
    :return: The expanded scores.
    """
    scoresx = np.zeros(shape=scores.shape[0] + (ccl_max * 2), dtype="float64")
    scoresx[ccl_max: -ccl_max] = scores
    scoresx[0: ccl_max] = 1 if e1 else 0
    scoresx[-ccl_max: -1] = 1 if e2 else 0
    return scoresx



def _calc_smu_moving(score: NDArray, mu_ds: int) -> NDArray:
    """
    Calculate smu moving based on score and down-sampled mu.

    :param score: The score.
    :param mu_ds: The mu_ds.
    :return: The calculated smu.
    """
    smu_fwd = bn.move_sum(score, window=mu_ds, min_count=1)
    smu_rev = bn.move_sum(score[::-1], window=mu_ds, min_count=1)
    smu = np.stack((smu_fwd, smu_rev))
    return smu



def _calc_benefit_moving(score: NDArray, ccl_ds: NDArray) -> NDArray:
    """
    Calculate benefit moving based on score and ccl_ds.

    :param score: Fragment scores.
    :param ccl_ds: Down-sampled read length dist array.
    :return: The calculated benefit.
    """
    score_rev = score[::-1]
    benefit = np.zeros(shape=(2, score.shape[0]), dtype="float64")
    perc = np.arange(0.1, 1.1, 0.1)[::-1]
    assert perc.shape == ccl_ds.shape
    for i in range(ccl_ds.shape[0]):
        ben_fwd = bn.move_sum(score, window=ccl_ds[i], min_count=1)[ccl_ds[i]: -1]
        ben_rev = bn.move_sum(score_rev, window=ccl_ds[i], min_count=1)[ccl_ds[i]: -1]
        benefit[0, 0: -ccl_ds[i] - 1] += ben_fwd * perc[i]
        benefit[1, ccl_ds[i]: -1] += ben_rev[::-1] * perc[i]
    return benefit




def benefit_bins(benefit: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Group benefit into bins of similar values using binary exponent. Used to find acceptance threshold

    :param benefit: positional benefit array.
    :return: The benefit bins and counts.
    """
    benefit_nz_ind = np.nonzero(benefit)
    benefit_flat_nz = benefit[benefit_nz_ind]
    # to make binary exponents work, normalise benefit values
    normaliser = np.max(benefit_flat_nz)
    benefit_flat_norm = benefit_flat_nz / normaliser
    mantissa, benefit_exponents = np.frexp(benefit_flat_norm)
    # count how often each exponent is present
    # absolute value because counting positive integers is quicker
    benefit_exponents_pos = np.abs(benefit_exponents)
    # multi-thread counting of exponents
    exponent_arrays = np.array_split(benefit_exponents_pos, 12)
    with TPexe(max_workers=12) as executor:
        exponent_counts = executor.map(np.bincount, exponent_arrays)
    exponent_counts = list(exponent_counts)
    # aggregate results from threads
    # target array needs to have the largest shape of the thread results
    max_exp = np.max([e.shape[0] for e in exponent_counts])
    bincounts = np.zeros(shape=max_exp, dtype='int')
    # sum up results from individual threads
    for exp in exponent_counts:
        bincounts[0:exp.shape[0]] += exp
    # filter empty bins
    exponents_unique = np.nonzero(bincounts)[0]
    # counts of the existing benefit exponents
    counts = bincounts[exponents_unique]
    # use exponents to rebuild benefit values
    benefit_bin = np.power(2.0, -exponents_unique) * normaliser
    return benefit_bin, counts



