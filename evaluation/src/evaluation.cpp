#include <cstdlib>
#include <cmath>
#include <opencv2/features2d/features2d.hpp>
#include <include/evaluation.h>

using namespace std;
using namespace cv;

namespace autocalib {
namespace evaluation {

void SyntheticScene::TakeShot(const RigidCamera &camera, Rect viewport,
                              detail::ImageFeatures &features)
{
    Mat R_inv = camera.R().inv();
    Point3d origin = Mat(-R_inv * camera.T()).at<Point3d>(0, 0);

    vector<int> visible_points;
    features.keypoints.clear();

    Mat_<double> P = camera.P();
    for (size_t i = 0; i < points.size(); ++i) {
        Point3d pt = points[i];
        if (IsVisible(pt, origin)) {
            double x = P(0, 0) * pt.x + P(0, 1) * pt.y + P(0, 2) * pt.z + P(0, 3);
            double y = P(1, 0) * pt.x + P(1, 1) * pt.y + P(1, 2) * pt.z + P(1, 3);
            double z = P(2, 0) * pt.x + P(2, 1) * pt.y + P(2, 2) * pt.z + P(2, 3);
            Point2f kp(float(x / z), float(y / z));
            if (kp.x > (float)viewport.x && kp.x < float(viewport.width - 1) &&
                kp.y > (float)viewport.y && kp.y < float(viewport.height - 1))
            {
                visible_points.push_back(i);
                features.keypoints.push_back(KeyPoint(kp, 1.f));
            }
        }
    }

    features.descriptors.create(visible_points.size(), 1, CV_32S);
    for (size_t i = 0; i < visible_points.size(); ++i)
        features.descriptors.at<int>(i ,0) = visible_points[i];

    features.img_size = viewport.size();
}


SphereScene::SphereScene(int num_points, RNG &rng) {
    points.resize(num_points);
    for (int i = 0; i < num_points; ++i) {
        double phi = (double)rng * 2. * CV_PI;
        double psi = (double)rng * CV_PI;
        points[i].x = cos(phi) * sin(psi);
        points[i].y = sin(phi) * sin(psi);
        points[i].z = cos(psi);
    }
}


bool SphereScene::IsVisible(const Point3d &point, const Point3d &origin) const {
    return point.x * origin.x + point.y * origin.y + point.z * origin.z > 0;
}


CubeScene::CubeScene(int num_points, RNG &rng) {
    points.resize(num_points);
    for (int i = 0; i < num_points; ++i) {
        int j = abs((int)rng) % 3;
        points[i].x = (j == 0 ? abs((int)rng) % 2 : (double)rng) - 0.5;
        points[i].y = (j == 1 ? abs((int)rng) % 2 : (double)rng) - 0.5;
        points[i].z = (j == 2 ? abs((int)rng) % 2 : (double)rng) - 0.5;
    }
}


bool CubeScene::IsVisible(const Point3d &point, const Point3d &origin) const {
    Point3d dir = point - origin;
    double dist = sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    dir *= 1 / dist;

    double x, y, z, t;

    t = (-0.5 - origin.x) / dir.x;
    y = origin.y + t * dir.y;
    z = origin.z + t * dir.z;
    if (y > -0.5 && y < 0.5 && z > -0.5 && z < 0.5 && t > 0 && t < dist - 1e-6)
        return false;

    t = (0.5 - origin.x) / dir.x;
    y = origin.y + t * dir.y;
    z = origin.z + t * dir.z;
    if (y > -0.5 && y < 0.5 && z > -0.5 && z < 0.5 && t > 0 && t < dist - 1e-6)
        return false;

    t = (-0.5 - origin.y) / dir.y;
    x = origin.x + t * dir.x;
    z = origin.z + t * dir.z;
    if (x > -0.5 && x < 0.5 && z > -0.5 && z < 0.5 && t > 0 && t < dist - 1e-6)
        return false;

    t = (0.5 - origin.y) / dir.y;
    x = origin.x + t * dir.x;
    z = origin.z + t * dir.z;
    if (x > -0.5 && x < 0.5 && z > -0.5 && z < 0.5 && t > 0 && t < dist - 1e-6)
        return false;

    t = (-0.5 - origin.z) / dir.z;
    x = origin.x + t * dir.x;
    y = origin.y + t * dir.y;
    if (x > -0.5 && x < 0.5 && y > -0.5 && y < 0.5 && t > 0 && t < dist - 1e-6)
        return false;

    t = (0.5 - origin.z) / dir.z;
    x = origin.x + t * dir.x;
    y = origin.y + t * dir.y;
    if (x > -0.5 && x < 0.5 && y > -0.5 && y < 0.5 && t > 0 && t < dist - 1e-6)
        return false;

    return true;
}


void MatchSyntheticShots(const detail::ImageFeatures &f1, const detail::ImageFeatures &f2,
                         vector<DMatch> &matches)
{
    vector<DMatch> matches_;
    BruteForceMatcher<L2<int> > matcher;
    matcher.match(f1.descriptors, f2.descriptors, matches_);

    matches.clear();
    for (size_t i = 0; i < matches_.size(); ++i)
        if (f1.descriptors.at<int>(matches_[i].queryIdx) ==
            f2.descriptors.at<int>(matches_[i].trainIdx))
            matches.push_back(matches_[i]);
}


void CreateImage(const detail::ImageFeatures &features, OutputArray image) {
    Mat &img_ = image.getMatRef();
    img_.create(features.img_size, CV_8U);
    img_.setTo(0);
    for (size_t i = 0; i < features.keypoints.size(); ++i)
        circle(img_, features.keypoints[i].pt, 1, Scalar::all(255), 2);
}

} // namespace evaluation
} // namespace autocalib
