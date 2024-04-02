import numpy as np
from scipy import stats
import math
from typing import List, Any, Tuple


# ToDo: Add type annotation to array?
def esd(data: np.ndarray, n_outliers: int, alpha: float = 0.05) -> np.ndarray:
    # ToDo: Force check for one dimensionality with ndim
    # ToDo: Force n_outliers > 0 because null hypothesis is zero outliers
    n_obs: int = data.size
    test_stats: np.ndarray = 


def _max_test_stat(data: np.ndarray) -> Tuple[float, int]:
    # This logic requires the data to always stay in the same order
    mean: float = data.mean()
    std: float = data.std()
    dists: np.ndarray = np.abs(data - mean)
    max_idx: int = dists.argmax()
    stat: float = dists[max_idx] / std
    return data[max_idx], max_idx


def _test_stats(data: np.ndarray, n_outliers: int) -> np.ndarray:
    # This is a performance bottle neck, but given the iteration here
    # it would be a trick to parallelize this computation
    # Replace this Any with a Numeric type
    stats: List[Any] = []
    for i in range(1, n_outliers + 1):
        ts, ts_idx = _max_test_stat(data)
        # Not a huge fan of the extensive use of mutability here
        stats.append(ts)
        data: np.ndarray = np.delete(data, ts_idx)
    return np.ndarray(stats)


# Use numba on this to vectorize it?
def _critical_val(test_stat_idx: float, n_obs: int, alpha: float) -> float:
    ptile: float = 1 - (alpha / (2 * (n_obs - test_stat_idx + 1)))
    dof: float = n_obs - test_stat_idx - 1
    perc_pt: float = stats.t.ppf(ptile, dof)
    numerator: float = (n_obs - test_stat_idx) * perc_pt
    a: float = dof + perc_pt**2
    b: float = n_obs - test_stat_idx + 1
    denominator: float = math.sqrt(a * b)
    return numerator / denominator


