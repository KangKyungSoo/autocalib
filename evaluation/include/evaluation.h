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
      * Descriptor of each returned feature is its pre-image point index from point cloud.
      *
      * \param camera Rigid camera parameters
      * \param viewport Viewing region
      * \param img Output image pointer
      * \param features Result image features
      */
    void TakeShot(const RigidCamera &camera, cv::Rect viewport, cv::Mat &img,
                  cv::detail::ImageFeatures &features)
        { TakeShotImpl(camera, viewport, true, img, features); }

    /** Takes a shot of the scene.
      *
      * \param camera Rigid camera parameters
      * \param viewport Viewing region
      * \param features Result image features
      * \see TakeShot()
      */
    void TakeShot(const RigidCamera &camera, cv::Rect viewport,
                  cv::detail::ImageFeatures &features)
    {
        cv::Mat mock;
        TakeShotImpl(camera, viewport, false, mock, features);
    }

protected:

    /** Checks point visibility.
      *
      * \return true if the point is visible from the given origin, false otherwise */        
    virtual bool IsVisible(const cv::Point3d &point, const cv::Point3d &origin) const = 0;

private:
    void TakeShotImpl(const RigidCamera &camera, cv::Rect viewport, bool create_img, cv::Mat &img,
                      cv::detail::ImageFeatures &features);
};


/** Describes a synthetic sphere scene.
  *
  * Created sphere has unit radius and center in the origin.
  */
class SphereScene : public SyntheticScene {
public:

    // TODO put rng param into base class
    /** Creates sphere scene.
      *
      * \param num_points Number of points on the sphere
      * \param rng Pseudo random number generator
      */
    SphereScene(int num_points, cv::RNG &rng);

private:
    virtual bool IsVisible(const cv::Point3d &point, const cv::Point3d &origin) const;
};


/** Matches two synthetic scene shots.
  *
  * \param f0 First shot features
  * \param f1 Second shot features
  * \param matches Found matches
  */
void MatchSyntheticShots(const cv::detail::ImageFeatures &f1, const cv::detail::ImageFeatures &f2,
                         std::vector<cv::DMatch> &matches);

} // namespace autocalib

#endif // AUTOCALIB_EVALUATION_H_
