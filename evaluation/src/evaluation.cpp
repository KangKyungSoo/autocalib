#include <cstdlib>
#include <cmath>
#include <opencv2/features2d/features2d.hpp>
#include <include/evaluation.h>

using namespace std;
using namespace cv;

namespace autocalib {
namespace evaluation {

    void PointCloudScene::TakeShot(const RigidCamera &camera, Rect viewport,
                                   detail::ImageFeatures &features)
    {
        Mat R_inv = camera.R().inv();
        Point3d origin = Mat(-R_inv * camera.T()).at<Point3d>(0, 0);

        vector<int> visible_points;
        features.keypoints.clear();

        Mat_<double> P = camera.P();
        for (size_t i = 0; i < points_.size(); ++i) {
            Point3d pt = points_[i];
            if (IsVisible(pt, origin)) {
                Point3d pt_ = TransformRigid(pt, R_, T_);
                pt.x = P(0, 0) * pt_.x + P(0, 1) * pt_.y + P(0, 2) * pt_.z + P(0, 3);
                pt.y = P(1, 0) * pt_.x + P(1, 1) * pt_.y + P(1, 2) * pt_.z + P(1, 3);
                pt.z = P(2, 0) * pt_.x + P(2, 1) * pt_.y + P(2, 2) * pt_.z + P(2, 3);
                Point2f kp(float(pt.x / pt.z), float(pt.y / pt.z));
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
            features.descriptors.at<int>(i, 0) = visible_points[i];

        features.img_size = viewport.size();
    }


    void CompositeScene::TakeShot(const RigidCamera &camera, Rect viewport,
                                  detail::ImageFeatures &features)
    {
        vector<detail::ImageFeatures> all_features(scenes_.size());
        size_t total_num_keypoints = 0;

        for (size_t i = 0; i < scenes_.size(); ++i) {
            scenes_[i]->TakeShot(camera, viewport, all_features[i]);
            total_num_keypoints += all_features[i].keypoints.size();
        }

        features.img_size = viewport.size();
        features.keypoints.resize(total_num_keypoints);
        features.descriptors.create(total_num_keypoints, 1, CV_32S);

        int keypoint_idx = 0;
        int descr_offset = 0;
        for (size_t i = 0; i < all_features.size(); ++i) {
            for (size_t j = 0; j < all_features[i].keypoints.size(); ++j) {
                features.keypoints[keypoint_idx] = all_features[i].keypoints[j];
                features.descriptors.at<int>(keypoint_idx, 0) =
                        all_features[i].descriptors.at<int>(j, 0) + descr_offset;
                keypoint_idx++;
            }
            descr_offset += scenes_[i]->pointCount();
        }
    }


    SphereScene::SphereScene(int num_points, RNG &rng) {
        points_.resize(num_points);
        for (int i = 0; i < num_points; ++i) {
            double phi = (double)rng * 2. * CV_PI;
            double psi = (double)rng * CV_PI;
            points_[i].x = cos(phi) * sin(psi);
            points_[i].y = sin(phi) * sin(psi);
            points_[i].z = cos(psi);
        }
    }


    bool SphereScene::IsVisible(const Point3d &point, const Point3d &origin) const {
        Point3d origin_ = TransformRigid(origin, R_.t(), -R_.t() * T_);
        return point.x * origin_.x + point.y * origin_.y + point.z * origin_.z > 0;
    }


    CubeScene::CubeScene(int num_points, RNG &rng) {
        points_.resize(num_points);
        for (int i = 0; i < num_points; ++i) {
            int j = abs((int)rng) % 3;
            points_[i].x = (j == 0 ? abs((int)rng) % 2 : (double)rng) - 0.5;
            points_[i].y = (j == 1 ? abs((int)rng) % 2 : (double)rng) - 0.5;
            points_[i].z = (j == 2 ? abs((int)rng) % 2 : (double)rng) - 0.5;
        }
    }


    bool CubeScene::IsVisible(const Point3d &point, const Point3d &origin) const {
        Point3d origin_ = TransformRigid(origin, R_.t(), -R_.t() * T_);
        Point3d dir = point - origin_;
        double dist = sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        dir *= 1 / dist;

        double x, y, z, t;

        t = (-0.5 - origin_.x) / dir.x;
        y = origin_.y + t * dir.y;
        z = origin_.z + t * dir.z;
        if (y > -0.5 && y < 0.5 && z > -0.5 && z < 0.5 && t > 0 && t < dist - 1e-6)
            return false;

        t = (0.5 - origin_.x) / dir.x;
        y = origin_.y + t * dir.y;
        z = origin_.z + t * dir.z;
        if (y > -0.5 && y < 0.5 && z > -0.5 && z < 0.5 && t > 0 && t < dist - 1e-6)
            return false;

        t = (-0.5 - origin_.y) / dir.y;
        x = origin_.x + t * dir.x;
        z = origin_.z + t * dir.z;
        if (x > -0.5 && x < 0.5 && z > -0.5 && z < 0.5 && t > 0 && t < dist - 1e-6)
            return false;

        t = (0.5 - origin_.y) / dir.y;
        x = origin_.x + t * dir.x;
        z = origin_.z + t * dir.z;
        if (x > -0.5 && x < 0.5 && z > -0.5 && z < 0.5 && t > 0 && t < dist - 1e-6)
            return false;

        t = (-0.5 - origin_.z) / dir.z;
        x = origin_.x + t * dir.x;
        y = origin_.y + t * dir.y;
        if (x > -0.5 && x < 0.5 && y > -0.5 && y < 0.5 && t > 0 && t < dist - 1e-6)
            return false;

        t = (0.5 - origin_.z) / dir.z;
        x = origin_.x + t * dir.x;
        y = origin_.y + t * dir.y;
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


    Mat CreateImage(const detail::ImageFeatures &features) {
        Mat img(features.img_size, CV_8U);
        img.setTo(0);

        for (size_t i = 0; i < features.keypoints.size(); ++i)
            circle(img, features.keypoints[i].pt, 1, Scalar::all(255), 1);

        return img;
    }

} // namespace evaluation
} // namespace autocalib
