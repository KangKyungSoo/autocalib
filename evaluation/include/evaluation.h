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

    /** Checks point visibility.
      *
      * \return true if the point is visible from the given origin, false otherwise */
    virtual bool IsVisible(const cv::Point3d &point, const cv::Point3d &origin) const = 0;
};


/** Describes a synthetic sphere scene. */
class SphereScene : public SyntheticScene {
public:

    /** Creates sphere scene.
      *
      * \param num_points Number of points on the sphere
      * \param seed RNG seed, pass -1 if seeding isn't needed
      */
    SphereScene(int num_points, int seed = -1);

private:
    virtual bool IsVisible(const cv::Point3d &point, const cv::Point3d &origin) const;
};

} // namespace autocalib

#endif // AUTOCALIB_EVALUATION_H_
