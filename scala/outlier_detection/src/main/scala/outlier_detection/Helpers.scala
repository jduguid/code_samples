package outlier_detection

object Helpers {
  case class PotentialOutlier(
    value: Double,
    testStat: Double,
    index: Int
  )

  case class CriticalValue(
    numOutliers: Int,
    criticalValue: Double
  )

  def dropElemAt[A](data: Vector[A], idx: Int): Vector[A] = {
    val firstSlice: Vector[A] = data.slice(0, idx)
    val secondSlice: Vector[A] = data.slice(idx, data.size).tail
    firstSlice ++ secondSlice
  }
}
