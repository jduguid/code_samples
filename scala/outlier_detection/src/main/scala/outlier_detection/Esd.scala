package outlier_detection

import scala.math.{abs, pow, sqrt}
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics
import org.apache.commons.math3.distribution.TDistribution
import outlier_detection.HelperClasses.CriticalValue
import spire.std.double


object Esd {
  def esdTest[A](data: Array[A], nOutliers: Int, alpha: Double = 0.05)(implicit num: Numeric[A]): Int = {
    val nObs: Int = data.size
    val tss: Array[Double] = testStats(data, nOutliers)
    val critVals: Array[CriticalValue] = criticalValues(nOutliers, nObs, alpha)
    val testsPasssed: Array[Int] = {
      tss.zip(critVals).filter(
        tuple => {
          tuple._1 > tuple._2.criticalValue
        }
      )
      .map(tuple => tuple._2.numOutliers)
    }
    if (testsPasssed.isEmpty) {0} else {testsPasssed.max}
  }

  private def maxTestStat(data: Array[Double]): Double = {
    val descStats: DescriptiveStatistics = new DescriptiveStatistics(data)
    val dists: Array[Double] = data.map(i => abs(i - descStats.getMean))
    val stat: Double = dists.max / descStats.getStandardDeviation
    stat
  }
 
  private def testStats[A](data: Array[A], nOutliers: Int)(implicit num: Numeric[A]): Array[Double] = {
    val sortedData: Array[Double] = data.map(i => num.toDouble(i)).sortWith((a, b) => a > b)
    val maxes: Array[Double] = (1 to nOutliers)
      .foldLeft(Array.empty[Double])(
        (acc, i) => {
          val max: Double = maxTestStat(sortedData.drop(i))
          acc :+ max
        }
      )
      maxes
  }

  private def criticalValue(testStatIdx: Int, nObs: Int, alpha: Double): Double = {
    val ptile: Double = (1 - (alpha / (2 * (nObs - testStatIdx + 1)))).toDouble
    val dof: Double = (nObs - testStatIdx - 1).toDouble
    // Might need to change this to n - 1
    val percPt: Double = new TDistribution(1).inverseCumulativeProbability(alpha)
    val numerator: Double = (nObs - testStatIdx) * percPt
    val a: Double = dof + pow(percPt, 2)
    val b: Double = (nObs - testStatIdx + 1).toDouble
    val denominator: Double = sqrt(a * b)
    numerator / denominator
  }

  private def criticalValues(nTestStats: Int, nObs: Int, alpha: Double): Array[CriticalValue] = {
    (1 to nTestStats).map(
      i => {
        val cv: Double = criticalValue(i, nObs, alpha)
        CriticalValue(i, cv)
      }
    ).toArray
  }
}