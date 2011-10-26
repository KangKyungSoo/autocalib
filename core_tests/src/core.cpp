#include "precomp.h"
#include <core/include/core.h>
#include <evaluation/include/evaluation.h>

using namespace std;
using namespace cv;
using namespace autocalib;
using namespace autocalib::evaluation;


TEST(Anitdiag, SquareIsUnit) {
    Mat A = Antidiag(3, 3, CV_64F);

    ASSERT_TRUE(A.size() == Size(3, 3));
    ASSERT_EQ(CV_64F, A.type());
    ASSERT_LE(norm(Mat::eye(3, 3, CV_64F), A * A), 1e-6);
}


TEST(DecomposeCholesky, CanDecomposeSmallMatrix) {
    Mat_<double> L = Mat::zeros(3, 3, CV_64F);
    L(0, 0) = 1;
    L(1, 0) = 2; L(1, 1) = 3;
    L(2, 0) = 4; L(2, 1) = 5; L(2, 2) = 6;

    Mat dst = DecomposeCholesky(L * L.t());

    ASSERT_TRUE(!dst.empty());
    ASSERT_LT(norm(dst, L, NORM_INF), 1e-6);
}


TEST(DecomposeCholesky, CanNotDecomposeNegativeDefiniteMatrix) {
    Mat_<double> L = Mat::zeros(3, 3, CV_64F);
    L(0, 0) = 1;
    L(1, 0) = 2; L(1, 1) = 3;
    L(2, 0) = 4; L(2, 1) = 5; L(2, 2) = 6;

    ASSERT_TRUE(DecomposeCholesky(-L * L.t()).empty());
}


TEST(DecomposeUUt, CanDecomposeSmallMatrix) {
    Mat_<double> U = Mat::zeros(3, 3, CV_64F);
    U(0, 0) = 1; U(0, 1) = 2; U(0, 2) = 3;
    U(1, 1) = 4; U(1, 2) = 5;
    U(2, 2) = 6;
   
    Mat dst = DecomposeUUt(U * U.t());

    ASSERT_TRUE(!dst.empty());
    ASSERT_LT(norm(dst, U, NORM_INF), 1e-3);
}


TEST(DltTraingulation, CanTriangluateSphere) {
    RNG rng(0);
    int num_points = 10;
    double max_angle = 0.1;
    Rect viewport = Rect(0, 0, 640, 480);

    Ptr<PointCloudScene> scene = new SphereScene(num_points, rng);

    Mat_<double> K = Mat::eye(3, 3, CV_64F);
    K(0, 0) = K(1, 1) = viewport.width + viewport.height;
    K(0, 2) = viewport.width * 0.5;
    K(1, 2) = viewport.height * 0.5;

    vector<RigidCamera> cameras(2);
    FeaturesCollection features_collection;

    for (int i = 0; i < 2; ++i) {
        Mat_<double> center = Mat::zeros(3, 1, CV_64F);
        center(0, 0) = i * 2 - 1; center(2, 0) = -10;
        cameras[i] = RigidCamera::LocalToWorld(K, Mat::eye(3, 3, CV_64F), center);
        Ptr<detail::ImageFeatures> features = new detail::ImageFeatures();
        scene->TakeShot(cameras[i], viewport, *features);
        features_collection[i] = features;
    }
    
    vector<DMatch> matches;
    MatchSyntheticShots(*(features_collection.find(0)->second),
                        *(features_collection.find(1)->second),
                        matches);

    Mat_<double> xy0(1, matches.size() * 2);
    Mat_<double> xy1(1, matches.size() * 2);
    const detail::ImageFeatures &f0 = *(features_collection.find(0)->second);
    const detail::ImageFeatures &f1 = *(features_collection.find(1)->second);
    for (size_t i = 0; i < matches.size(); ++i)  {
        xy0(0, 2 * i) = f0.keypoints[matches[i].queryIdx].pt.x;
        xy0(0, 2 * i + 1) = f0.keypoints[matches[i].queryIdx].pt.y;
        xy1(0, 2 * i) = f1.keypoints[matches[i].trainIdx].pt.x;
        xy1(0, 2 * i + 1) = f1.keypoints[matches[i].trainIdx].pt.y;
    }

    DltTriangulation dlt;

    Mat_<double> xyzw;
    dlt.triangulate(cameras[0], cameras[1], xy0, xy1, xyzw);

    for (size_t i = 0; i < matches.size(); ++i) {
        int point_idx = f0.descriptors.at<int>(matches[i].queryIdx);
        Point3d point_gold = scene->localPointAt(point_idx);
        Point3d point_found(xyzw(0, 4 * i) / xyzw(0, 4 * i + 3),
                            xyzw(0, 4 * i + 1) / xyzw(0, 4 * i + 3),
                            xyzw(0, 4 * i + 2) / xyzw(0, 4 * i + 3));
        ASSERT_NEAR(point_gold.x, point_found.x, 1e-6);
        ASSERT_NEAR(point_gold.y, point_found.y, 1e-6);
        ASSERT_NEAR(point_gold.z, point_found.z, 1e-6);
        //Mat_<double> P1_ = cameras[0].P();
        //Mat_<double> P2_ = cameras[1].P();
        //Point3d pt = point_found;
        //double x = P1_(0, 0) * pt.x + P1_(0, 1) * pt.y + P1_(0, 2) * pt.z + P1_(0, 3);
        //double y = P1_(1, 0) * pt.x + P1_(1, 1) * pt.y + P1_(1, 2) * pt.z + P1_(1, 3);
        //double z = P1_(2, 0) * pt.x + P1_(2, 1) * pt.y + P1_(2, 2) * pt.z + P1_(2, 3);
        //cout << xy0(0, 2 * i) << " " << xy0(0, 2 * i + 1) << " " << x / z << " " << y / z << endl;
        //pt = point_gold;
        //x = P1_(0, 0) * pt.x + P1_(0, 1) * pt.y + P1_(0, 2) * pt.z + P1_(0, 3);
        //y = P1_(1, 0) * pt.x + P1_(1, 1) * pt.y + P1_(1, 2) * pt.z + P1_(1, 3);
        //z = P1_(2, 0) * pt.x + P1_(2, 1) * pt.y + P1_(2, 2) * pt.z + P1_(2, 3);
        //cout << xy0(0, 2 * i) << " " << xy0(0, 2 * i + 1) << " " << x / z << " " << y / z << endl;
        //cout << point_gold << " " << point_found << " " << xyzw(0, 4 * i) << " " << xyzw(0, 4 * i + 1) << " " << xyzw(0, 4 * i + 2) << " " << xyzw(0, 4 * i + 3) << endl;
    }
}
