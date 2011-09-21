#include "precomp.h"
#include <include/core.h>

using namespace std;
using namespace cv;

namespace autocalib {

void CalibRotationalCamera(cv::InputArrayOfArrays keypoints,
                           const std::map<std::pair<int, int>, cv::Mat> &matches,
                           cv::InputOutputArray K,
                           cv::InputArray mask)
{
}

} // namespace autocalib
