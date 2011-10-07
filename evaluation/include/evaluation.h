#ifndef AUTOCALIB_EVALUATION_H_
#define AUTOCALIB_EVALUATION_H_

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <core/include/core.h>

namespace autocalib {
namespace evaluation {

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
      * \param viewport Viewing region
      * \param features Result image features
      */
    void TakeShot(const RigidCamera &camera, cv::Rect viewport,
                  cv::detail::ImageFeatures &features);

protected:

    /** Checks point visibility.
      *
      * \return true if the point is visible from the given origin, false otherwise */        
    virtual bool IsVisible(const cv::Point3d &point, const cv::Point3d &origin) const = 0;
};


/** Describes a synthetic sphere scene.
  *
  * Created sphere has unit radius and center in the origin.
  */
class SphereScene : public SyntheticScene {
public:

    /** Creates a sphere scene.
      *
      * \param num_points Number of points on the sphere
      * \param rng Pseudo random number generator
      */
    SphereScene(int num_points, cv::RNG &rng);

private:
    virtual bool IsVisible(const cv::Point3d &point, const cv::Point3d &origin) const;
};


/** Describes a synthetic cube scene.
  *
  * Creates cube has unit edge length and center in the origin.
  */
class CubeScene : public SyntheticScene {
public:

    /** Creates a cube scene.
      *
      * \param num_points Number of points on the cube
      * \param rng Pseudo random number generator
      */
    CubeScene(int num_points, cv::RNG &rng);

private:
    virtual bool IsVisible(const cv::Point3d &point, const cv::Point3d &origin) const;
};


/** Synthetic scenes factory. */
class SyntheticSceneCreator {
public:
    virtual ~SyntheticSceneCreator() {}

    /** Creates a synthetic scene.
      *
      * \param num_points Number of points
      * \param rng Pseudo random number generator
      */
    virtual cv::Ptr<SyntheticScene> Create(int num_points, cv::RNG &rng) = 0;
};


class SphereSceneCreator : public SyntheticSceneCreator {
public:
    virtual cv::Ptr<SyntheticScene> Create(int num_points, cv::RNG &rng) {
        return new SphereScene(num_points, rng);
    }
};


class CubeSceneCreator : public SyntheticSceneCreator {
public:
    virtual cv::Ptr<SyntheticScene> Create(int num_points, cv::RNG &rng) {
        return new CubeScene(num_points, rng);
    }
};


/** Matches two synthetic scene shots.
  *
  * \param f1 First shot features
  * \param f2 Second shot features
  * \param matches Found matches
  */
void MatchSyntheticShots(const cv::detail::ImageFeatures &f1, const cv::detail::ImageFeatures &f2,
                         std::vector<cv::DMatch> &matches);

/** Creates an image from image features.
  *
  * \param features Image features
  * \param image Result image
  */
void CreateImage(const cv::detail::ImageFeatures &features, cv::OutputArray image);

} // namespace evaluation
} // namespace autocalib

#endif // AUTOCALIB_EVALUATION_H_
