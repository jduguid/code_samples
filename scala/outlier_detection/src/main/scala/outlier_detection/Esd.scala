package outlier_detection

import scala.annotation.tailrec
import scala.math.{abs, pow, sqrt}
import breeze.stats.{mean, stddev}
import org.apache.commons.math3.distribution.TDistribution
import outlier_detection.Helpers.{CriticalValue, PotentialOutlier, dropElemAt}


object Esd {
  def esdTest[A](data: Vector[A], nOutliers: Int, alpha: Double = 0.05)(implicit num: Numeric[A]): Int = {
    val nObs: Int = data.size
    val doubleVals: Vector[Double] = data.map(i => num.toDouble(i))
    val tss: Vector[PotentialOutlier] = testStats(doubleVals, nOutliers)
    val critVals: Vector[CriticalValue] = criticalValues(nOutliers, nObs, alpha)
    val testsPasssed: Vector[Int] = {
      tss.zip(critVals).filter(
        tuple => {
          tuple._1.testStat > tuple._2.criticalValue
        }
      )
      .map(tuple => tuple._2.numOutliers)
    }
    if (testsPasssed.isEmpty) {0} else {testsPasssed.max}
  }

  private def maxTestStat(data: Vector[Double]): PotentialOutlier = {
    val std: Double = stddev(data)
    val avg: Double = mean(data)
    val dists: Vector[Double] = data.map(i => abs(i - avg))
    val maxDist: Double = dists.max
    val idxOfMax: Int = dists.indexOf(maxDist)
    val testStat: Double = maxDist / std
    val value: Double = data(idxOfMax)
    PotentialOutlier(value, testStat, idxOfMax)
  }

  private def testStats(data: Vector[Double], nOutliers: Int): Vector[PotentialOutlier] = {
    @tailrec
    def loop(data: Vector[Double], iterations: List[Int], maxes: Vector[PotentialOutlier]): Vector[PotentialOutlier] = {
      iterations match {
        case Nil => maxes
        case i :: iters => {
          val maxStat: PotentialOutlier = maxTestStat(data)
          val nextData: Vector[Double] = Helpers.dropElemAt(data, maxStat.index)
          loop(nextData, iters, maxes :+ maxStat)
        } 
      }  
    }
    loop(data, List.range(0, nOutliers), Vector.empty[PotentialOutlier])
  }

  private def criticalValue(testStatIdx: Int, nObs: Int, alpha: Double): Double = {
    val ptile: Double = 1.0 - (alpha / (2.0 * (nObs - testStatIdx + 1).toDouble))
    val dof: Int = nObs - testStatIdx - 1
    val percPt: Double = new TDistribution(dof).inverseCumulativeProbability(ptile)
    val numerator: Double = (nObs - testStatIdx).toDouble * percPt
    val a: Double = dof.toDouble + pow(percPt, 2)
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

  private def stdDev[A](data: Vector[A])(implicit num: Numeric[A]): Double = {
    val doubles: Vector[Double] = data.map(i => num.toDouble(i))
    val avg: Double = mean(doubles)
    val squaredDevs: Vector[Double] = doubles.map(i => pow(i - avg, 2))
    val variance: Double = mean(squaredDevs)
    sqrt(variance)
  }
}
