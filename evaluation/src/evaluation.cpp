#include <cstdlib>
#include <cmath>
#include <opencv2/features2d/features2d.hpp>
#include <include/evaluation.h>

using namespace std;
using namespace cv;

namespace autocalib {

Ptr<detail::ImageFeatures> SyntheticScene::TakeShot(const RigidCamera &camera, Rect viewport,
                                                    Mat *img)
{
    Mat R_inv = camera.R().inv();
    Point3d origin = Mat(-R_inv * camera.T()).at<Point3d>(0, 0);

    vector<int> visible_points;
    Ptr<detail::ImageFeatures> features = new detail::ImageFeatures;

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
                features->keypoints.push_back(KeyPoint(kp, 1.f));
            }
        }
    }

    features->descriptors.create(visible_points.size(), 1, CV_32S);
    for (size_t i = 0; i < visible_points.size(); ++i)
        features->descriptors.at<int>(i ,0) = visible_points[i];

    features->img_size = viewport.size();

    if (img) {
        img->create(viewport.size(), CV_8U);
        img->setTo(0);
        for (size_t i = 0; i < visible_points.size(); ++i)
            circle(*img, features->keypoints[i].pt, 1, Scalar::all(255), 2);
    }

    return features;
}


SphereScene::SphereScene(int num_points, int seed) {
    if (seed != -1)
        srand(seed);
    points.resize(num_points);
    for (int i = 0; i < num_points; ++i) {
        double phi = rand() / (double)RAND_MAX * 2. * CV_PI;
        double psi = rand() / (double)RAND_MAX * CV_PI;
        points[i].x = cos(phi) * sin(psi);
        points[i].y = sin(phi) * sin(psi);
        points[i].z = cos(psi);
    }
}


bool SphereScene::IsVisible(const Point3d &point, const Point3d &origin) const {
    return point.x * origin.x + point.y * origin.y + point.z * origin.z > 0;
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

} // namespace autocalib
