package outlier_detection

import outlier_detection.Esd
import outlier_detection.HelperClasses.CriticalValue
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.PrivateMethodTester
import org.scalatest.PrivateMethodTester.PrivateMethod
import scala.io.Source

class TestEsdComponents extends AnyFunSuite with PrivateMethodTester{
  val testData: Vector[Double] = {
    val buffer: Source = Source.fromResource("rosner_data.csv")
    val data: Vector[Double] = buffer
      .getLines()
      .foldLeft(Vector.empty[Double])((acc, i) => acc :+ i.toDouble)
    buffer.close
    data
  }  

  test("There should be 3 outliers in the NIST-provided data from Rosner (1983)") {
    val rosnerOutliers: Int = 3
    val testOutliers: Int = Esd.esdTest(testData, 10)
    assert(testOutliers == rosnerOutliers)
  }

  test("Esd.criticalValues should produce the same values laid out in NIST spec") {
    val criticalValues = PrivateMethod[Vector[CriticalValue]]('criticalValues)
    val rosnerCvs: Vector[Double] = Vector(
      3.158, 3.151, 3.143, 3.136, 3.128, 3.120, 3.111, 3.103, 3.094, 3.085
    )
    val testCvs: Vector[CriticalValue] = Esd invokePrivate criticalValues(10, testData.size, 0.05)
    val testCvDoubles: Vector[Double] = testCvs.map(i => i.criticalValue)
    assert(testCvDoubles == rosnerCvs)
  }

  test("Esd.testStats should produce the same values laid out in NIST spec") {
    val testStats = PrivateMethod[Vector[Double]]('testStats)
    val rosnerTss: Vector[Double] = Vector(
      3.118, 2.942, 3.179, 2.810, 2.815, 2.848, 2.279, 2.310, 2.101, 2.067
    )
    val testTss: Vector[Double] = Esd invokePrivate testStats(testData, 10)
    assert(testTss == rosnerTss)
  }
}
