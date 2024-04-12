from pathlib import Path
import numpy as np
import numpy.typing as npt
from typing import List, Union
from outlier_detection.esd import esd_test, _critical_vals, _test_stats



def get_test_data() -> npt.NDArray[np.float_]:
        repo_root: Path = Path(__file__).parents[3]
        data_path: Path = repo_root.joinpath("rosner_data.csv")
        rosner_data: npt.NDArray[np.float_] = np.genfromtxt(str(data_path), delimiter=",")
        return rosner_data
    

def acceptable_diff(a: List[np.float_], b: List[np.float_], thresh: float) -> bool:
    diffs: List[np.number] = [abs(i - j) for i, j in zip(a, b)]
    diffs_acceptable: List[np.bool_] = [d <= thresh for d in diffs]
    return all(diffs_acceptable)
    

class TestEsdComponents:
    """
    Replicates results of Rosner (1983) described in National Institute for Technology 
    and Standards guide at: https://www.itl.nist.gov/div898/handbook/eda/section3//eda35h3.htm

    The test data, critical values, and test statistics used in this test class can all be 
    found on that page.
    """
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
        rosner_cvs: List[np.float_] = [np.float_(i) for i in rosner_cvs]
        rosner_data: npt.NDArray[np.float_] = get_test_data()
        cvs: List[np.float_] = [np.round(v.critical_value, 3) for v in _critical_vals(10, rosner_data.size, 0.05)]
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
        rosner_tss: List[np.float_] = [np.float_(i) for i in rosner_tss]
        rosner_data: npt.NDArray[np.float_] = get_test_data()
        tss: List[np.float_] = [np.round(t.test_stat, 3) for t in _test_stats(rosner_data, 10)]
        assert acceptable_diff(rosner_tss, tss, 0.01)
