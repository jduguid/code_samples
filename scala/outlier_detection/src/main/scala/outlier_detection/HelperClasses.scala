package outlier_detection

object HelperClasses {
  case class PotentialOutlier[A: Numeric](
    value: A,
    testStat: Double,
    index: Int
  )

  case class CriticalValue(
    numOutliers: Int,
    criticalValue: Double
  )
}
