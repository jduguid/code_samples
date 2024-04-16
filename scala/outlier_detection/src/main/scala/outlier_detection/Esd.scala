package outlier_detection

import scala.math.{abs, pow, sqrt}
import breeze.stats.{mean, stddev}
import org.apache.commons.math3.distribution.TDistribution
import outlier_detection.HelperClasses.CriticalValue


object Esd {
  def esdTest[A](data: Vector[A], nOutliers: Int, alpha: Double = 0.05)(implicit num: Numeric[A]): Int = {
    val nObs: Int = data.size
    val doubleVals: Vector[Double] = data.map(i => num.toDouble(i))
    val tss: Vector[Double] = testStats(doubleVals, nOutliers)
    val critVals: Vector[CriticalValue] = criticalValues(nOutliers, nObs, alpha)
    val testsPasssed: Vector[Int] = {
      tss.zip(critVals).filter(
        tuple => {
          tuple._1 > tuple._2.criticalValue
        }
      )
      .map(tuple => tuple._2.numOutliers)
    }
    if (testsPasssed.isEmpty) {0} else {testsPasssed.max}
  }

  private def maxTestStat(data: Vector[Double]): Double = {
    val avg: Double = mean(data)
    val std: Double = stddev(data)
    val dists: Vector[Double] = data.map(i => abs(i - avg))
    val stat: Double = dists.max / std
    stat
  }
 
  private def testStats(data: Vector[Double], nOutliers: Int): Vector[Double] = {
    val sortedData: Vector[Double] = data.sortWith((a, b) => a > b)
    val maxes: Vector[Double] = (1 to nOutliers)
      .foldLeft(Vector.empty[Double])(
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

  private def criticalValues(nTestStats: Int, nObs: Int, alpha: Double): Vector[CriticalValue] = {
    (1 to nTestStats).map(
      i => {
        val cv: Double = criticalValue(i, nObs, alpha)
        CriticalValue(i, cv)
      }
    ).toVector
  }
}
