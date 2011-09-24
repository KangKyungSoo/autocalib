#ifndef AUTOCALIB_EVALUATION_H_
#define AUTOCALIB_EVALUATION_H_

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <core/include/core.h>

namespace autocalib {

/** Describes a point cloud. */
class PointCloud {
public:
    std::vector<cv::Point3d> points;
};


/** Synthetic scene base class. */
class SyntheticScene : public PointCloud {
public:
    virtual ~SyntheticScene() {}

    /** Takes a shot of the scene.
      *
      * \param camera Rigid camera parameters
      * \param roi Result image ROI
      * \param img Output image pointer, pass 0 if image isn't needed
      * \return Result image features
      */
    cv::Ptr<cv::detail::ImageFeatures> TakeShot(const RigidCamera &camera, cv::Rect roi,
                                                cv::Mat *img = 0);

protected:

    /** \return true if the point is visible from the given origin. */
    virtual bool IsVisible(const cv::Point3d &point, const cv::Point3d &origin) const = 0;
};




} // namespace autocalib

#endif // AUTOCALIB_EVALUATION_H_
