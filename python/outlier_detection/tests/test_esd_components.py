from importlib import resources
import numpy as np
import numpy.typing as npt
from typing import List
from outlier_detection.esd import esd_test, _critical_vals, _test_stats, Numeric


def get_test_data() -> npt.NDArray[np.float_]:
        with resources.path("outlier_detection.tests", "rosner_data.csv") as path:
            rosner_data: npt.NDArray[np.float_] = np.genfromtxt(path, delimiter=",")
        return rosner_data
    

def acceptable_diff(a: List[Numeric], b: List[Numeric], thresh: Numeric) -> bool:
    diffs: List[Numeric] = [abs(i - j) for i, j in zip(a, b)]
    diffs_acceptable: List[bool] = [d <= thresh for d in diffs]
    return all(diffs_acceptable)
    

class TestEsdComponents:
    def test_rosner_result(self):
        rosner_outliers: int = 3
        rosner_data: npt.NDArray[np.float_] = get_test_data()
        assert esd_test(rosner_data, 10) == rosner_outliers

    def test_rosner_crit_vals(self):
        rosner_cvs: List[float] = [
            3.158,
            3.151,
            3.143,
            3.136,
            3.128,
            3.120,
            3.111,
            3.103,
            3.094,
            3.085,
        ]
        rosner_data: npt.NDArray[np.float_] = get_test_data()
        cvs: List[float] = [round(v.critical_value, 3) for v in _critical_vals(10, rosner_data.size, 0.05)]
        assert acceptable_diff(rosner_cvs, cvs, 0.01)

    def test_rosner_test_stats(self):
        rosner_tss: List[float] = [
            3.118,
            2.942,
            3.179,
            2.810,
            2.815,
            2.848,
            2.279,
            2.310,
            2.101,
            2.067,
        ]
        rosner_data: npt.NDArray[np.float_] = get_test_data()
        tss: List[float] = [round(t.test_stat, 3) for t in _test_stats(rosner_data, 10)]
        assert acceptable_diff(rosner_tss, tss, 0.01)
