import numpy as np
import numpy.typing as npt
from scipy import stats
import math
from typing import List, Union, Tuple
from dataclasses import dataclass


def esd_test(data: npt.NDArray[np.number], n_outliers: int, alpha: float = 0.05) -> int:
    """
    Returns the number of outliers in a univariate, approximately normal data set.

    The Generalized Extreme Studentized Deviate (ESD) test as defined in Rosner (1983) 
    tests the hypothesis that there are up to `n_outliers` in the data set against the 
    null hypothesis that there are no outliers in the data set. The data must be 
    univariate and follow an approximately normal distribution.

    For more information on this test see the National Institute of Standards and
    Technology guide at `https://www.itl.nist.gov/div898/handbook/eda/section3//eda35h3.htm`.

    The original paper describing the procedure is: 
    Rosner, B. (1983). Percentage Points for a Generalized ESD Many-Outlier Procedure. Technometrics 25, 165–172. 
    
    :note: On performance. This function scales approximately linearly with the amount of data to be tested 
           and with the number of outliers to be tested. For example, testing for 10 outliers in a 
           series of 10 million could take 1 second, while testing for 10 outliers in a series 
           of 100 million could take 10 seconds. Similarly, testing for 100 outliers in a series of 10 million 
           could take 10 seconds as opposed to 1 second when testing for 10 outliers in the same series.  
    :param data: A 1-d numpy array of numeric data
    :param n_outliers: Maximum number of outliers to test for
    :param alpha: Confidence level of hypothesis tests
    """
    if data.ndim != 1:
        raise ValueError("Data must be one dimensional")
    if n_outliers == 0:
        raise ValueError(
            "Number of outliers must be greater than 0. Null hypothesis being tested for is that there are zero outliers"
        )
    n_obs: int = data.size
    ps: List[PotentialOutlier] = _test_stats(data, n_outliers)
    cvs: List[CriticalValue] = _critical_vals(n_outliers, n_obs, alpha)
    zipped_vals: List[Tuple[PotentialOutlier, CriticalValue]] = list(zip(ps, cvs))
    tests_passed: List[int] = [
        v.num_outliers for p, v in zipped_vals if p.test_stat > v.critical_value
    ]
    if tests_passed == []:
        num_outliers: int = 0
    else:
        num_outliers: int = max(tests_passed)
    return num_outliers


@dataclass
class PotentialOutlier:
    value: np.number
    test_stat: np.float_
    index: np.intp


def _max_test_stat(data: npt.NDArray[np.number]) -> PotentialOutlier:
    "Private function to calculate test stats following Rosner (1983)"
    # Testing the code with cProfile indicated that this function is the costliest in 
    # terms of performance. Because this function already uses fast numpy built-ins here,
    # optimizing this function further might be too costly in terms of current developer
    # time and future maintainability/readability.
    dists: npt.NDArray[np.number] = np.abs(data - data.mean())
    max_idx: np.intp = dists.argmax()
    stat: np.float_ = dists[max_idx] / data.std(ddof=1)
    return PotentialOutlier(data[max_idx], stat, max_idx)


def _test_stats(
    data: npt.NDArray[np.number], n_outliers: int
) -> List[PotentialOutlier]:
    "Calculate all test stats maximizing distance from mean, removing, then repeating"
    # cProfile also indicates this function could be a place for optimization. This is
    # because it calls the most expensive per-call function in the routine - 
    # _max_test_stat - n times. However, because of the ESD algorithm is iterative and  
    # each iteration requires information from the previous iteration, it may be quite 
    # tricky to optimize this function (e.g. through parallelization).
    potential_outliers: List[PotentialOutlier] = []
    for i in range(1, n_outliers + 1):
        # This max then append pattern essentially enforces the order required 
        # for correctness.
        p: PotentialOutlier = _max_test_stat(data)
        # I am not a huge fan of the use of mutability here, but am trading off 
        # safety in design for readability/maintainability here.
        potential_outliers.append(p)
        data: np.ndarray = np.delete(data, p.index)
    return potential_outliers


@dataclass
class CriticalValue:
    num_outliers: int
    critical_value: np.float_


def _critical_val(test_stat_idx: int, n_obs: int, alpha: float) -> np.float_:
    "Calculates critical values based on T-distribution following Rosner (1983)"
    ptile: float = 1 - (alpha / (2 * (n_obs - test_stat_idx + 1)))
    dof: int = n_obs - test_stat_idx - 1
    perc_pt: np.float_ = stats.t.ppf(ptile, dof)
    numerator: np.float_ = (n_obs - test_stat_idx) * perc_pt
    a: np.float_ = dof + perc_pt**2
    b: int = n_obs - test_stat_idx + 1
    denominator: float = math.sqrt(a * b)
    return numerator / denominator


def _critical_vals(n_test_stats: int, n_obs: int, alpha: float) -> List[CriticalValue]:
    "Calculates critical values for each value in n=1,2,...,n_outliers"
    crit_vals: List[CriticalValue] = []
    for i in range(1, n_test_stats + 1):
        cv: np.float_ = _critical_val(i, n_obs, alpha)
        crit_vals.append(CriticalValue(i, cv))
    return crit_vals
