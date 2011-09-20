#ifndef AUTOCALIB_CORE_H_
#define AUTOCALIB_CORE_H_

#include <vector>
#include <map>
#include <utility>
#include <opencv2/core/core.hpp>

namespace autocalib {

/** Calculates rotational camera intrinsics.
  *
  * See datails in Hartey R., Zisserman A., "Multiple View Geometry", 2nd ed., p. 482.
  *
  * \param kps Images keypoints vector.
  * \param matcher Mapping between view indices pairs and matches keypoint indices.
  * \param K Output camera intrinsics.
  */
void calibRotationalCamera(const std::vector<cv::Mat> &kps,
                           const std::map<std::pair<int, int>, cv::Mat> &matches,
                           cv::Mat &K);

} // namespace autocalib

#endif // AUTOCALIB_CORE_H_
