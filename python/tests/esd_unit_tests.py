from importlib import resources
import numpy as np
import numpy.typing as npt
from typing import List
from outlier_detection.esd import esd_test, _critical_vals, _test_stats


with resources.path("outlier_detection.tests", "rosner_data.csv") as path:
    test_data: npt.NDArray[np.float_] = np.genfromtxt(path, delimiter=",")


class EsdUnitTests:
    def test_rosner_result(self):
        rosner_outliers: int = 3
        assert esd_test(test_data, 10) == rosner_outliers

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
        cvs: List[float] = [
            round(v.critical_value, 3) for v in _critical_vals(10, test_data.size, 0.05)
        ]
        assert rosner_cvs == cvs

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
        tss: List[float] = [round(t.test_stat, 3) for t in _test_stats(test_data, 10)]
        assert rosner_tss == tss
