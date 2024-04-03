import numpy as np
import numpy.typing as npt
from scipy import stats
import math
from typing import List, Union, Tuple
from dataclasses import dataclass


Numeric = Union[int, float]

@dataclass
class PotentialOutlier:
    value: Numeric
    test_stat: float
    index: int

@dataclass
class CriticalValue:
    num_outliers: int
    critical_value: float


def esd_test(data: npt.NDArray[np.number], n_outliers: int, alpha: float = 0.05) -> int:
    if data.ndim != 1:
        raise ValueError("Data must be one dimensional")
    if n_outliers == 0:
        raise ValueError("Number of outliers must be greater than 0. Null hypothesis being tested for is that there are zero outliers")
    n_obs: int = data.size
    ps: List[PotentialOutlier] = _test_stats(data, n_outliers)
    cvs: List[CriticalValue] = _critical_vals(n_outliers, n_obs, alpha)
    zipped_vals: List[Tuple[PotentialOutlier, CriticalValue]] = list(zip(ps, cvs))
    tests_passed: List[int] = [v.num_outliers for p,v in zipped_vals if p.test_stat > v.critical_value]
    if tests_passed == []:
        num_outliers: int = 0
    else:
        num_outliers: int = max(tests_passed)
    return num_outliers


def _max_test_stat(data: npt.NDArray[np.number]) -> PotentialOutlier:
    dists: np.ndarray = np.abs(data - data.mean())
    max_idx: int = dists.argmax()
    stat: float = dists[max_idx] / data.std()
    return PotentialOutlier(data[max_idx], stat, max_idx)


def _test_stats(data: npt.NDArray[np.number], n_outliers: int) -> List[PotentialOutlier]:
    # This is a performance bottle neck, but given the iteration here
    # it would be a trick to parallelize this computation
    potential_outliers: List[PotentialOutlier] = []
    for i in range(1, n_outliers + 1):
        # This max then append pattern essentially enforces the order we want
        p: PotentialOutlier = _max_test_stat(data)
        # This append creates copies of the data which I presume need to be garbage collected
        # Fine for smaller data and lower number of outliers, but won't technically scale well
        potential_outliers.append(p)
        # Not a huge fan of the use of mutability here
        data: np.ndarray = np.delete(data, p.index)
    return potential_outliers


def _critical_val(test_stat_idx: int, n_obs: int, alpha: float) -> float:
    ptile: float = 1 - (alpha / (2 * (n_obs - test_stat_idx + 1)))
    dof: float = n_obs - test_stat_idx - 1
    perc_pt: float = stats.t.ppf(ptile, dof)
    numerator: float = (n_obs - test_stat_idx) * perc_pt
    a: float = dof + perc_pt**2
    b: float = n_obs - test_stat_idx + 1
    denominator: float = math.sqrt(a * b)
    return numerator / denominator


def _critical_vals(n_test_stats: int, n_obs: int, alpha: float) -> List[CriticalValue]:
    crit_vals: List[CriticalValue] = []
    for i in range(1, n_test_stats + 1):
        cv: float = _critical_val(i, n_obs, alpha)
        crit_vals.append(CriticalValue(i, cv))
    return crit_vals
