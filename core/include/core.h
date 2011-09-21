#ifndef AUTOCALIB_CORE_H_
#define AUTOCALIB_CORE_H_

#include <vector>
#include <map>
#include <utility>
#include <opencv2/core/core.hpp>

namespace autocalib {

/** Calculates rotational camera intrinsics.
  *
  * See details in Hartey R., Zisserman A., "Multiple View Geometry", 2nd ed., p. 482.
  *
  * \param keypoints Images keypoints
  * \param matches Mapping between view indices pairs and matches keypoint indices
  * \param K Output camera intrinsics
  * \param mask Mask identifying intrinsic parameters to be estimated
  */
void CalibRotationalCamera(cv::InputArrayOfArrays keypoints,
                           const std::map<std::pair<int, int>, cv::Mat> &matches,
                           cv::InputOutputArray K,
                           cv::InputArray mask = cv::Mat::ones(3, 3, CV_8U));

} // namespace autocalib

#endif // AUTOCALIB_CORE_H_
