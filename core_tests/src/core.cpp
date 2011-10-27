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


class DltTriangulationMetric : public testing::TestWithParam<Ptr<PointCloudSceneCreator> > { };

TEST_P(DltTriangulationMetric, CanTriangulate) {
    RNG rng(0);
    int num_points = 1000;
    double max_angle = 0.1;
    Rect viewport = Rect(0, 0, 640, 480);

    Ptr<PointCloudScene> scene = GetParam()->Create(num_points, rng);

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

        ASSERT_NEAR(point_gold.x, point_found.x, 1e-5);
        ASSERT_NEAR(point_gold.y, point_found.y, 1e-5);
        ASSERT_NEAR(point_gold.z, point_found.z, 1e-5);
    }
}

INSTANTIATE_TEST_CASE_P(StdSynthScenes,
                        DltTriangulationMetric,
                        testing::Values(new SphereSceneCreator(), new CubeSceneCreator()));


class DltTriangulationProjective : public testing::TestWithParam<Ptr<PointCloudSceneCreator> > { };

TEST_P(DltTriangulationProjective, CanTriangulate) {
    RNG rng(0);
    int num_points = 1000;
    double max_angle = 0.1;
    Rect viewport = Rect(0, 0, 640, 480);    

    Ptr<PointCloudScene> scene = GetParam()->Create(num_points, rng);

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

    Mat_<double> H = Mat::zeros(4, 4, CV_64F);
    while (abs(determinant(H)) < 1e-3)
        rng.fill(H, RNG::UNIFORM, -1, 1);
    H /= pow(abs(determinant(H)), 0.25);
    Mat_<double> H_inv = H.inv();

    DltTriangulation dlt;

    Mat_<double> P0 = cameras[0].P() * H_inv;
    Mat_<double> P1 = cameras[1].P() * H_inv;

    Mat_<double> xyzw;
    dlt.triangulate(ProjectiveCamera(P0), ProjectiveCamera(P1), xy0, xy1, xyzw);

    for (size_t i = 0; i < matches.size(); ++i) {
        double x0 = P0(0, 0) * xyzw(0, 4 * i) + P0(0, 1) * xyzw(0, 4 * i + 1) + P0(0, 2) * xyzw(0, 4 * i + 2) + P0(0, 3) * xyzw(0, 4 * i + 3);
        double y0 = P0(1, 0) * xyzw(0, 4 * i) + P0(1, 1) * xyzw(0, 4 * i + 1) + P0(1, 2) * xyzw(0, 4 * i + 2) + P0(1, 3) * xyzw(0, 4 * i + 3);
        double z0 = P0(2, 0) * xyzw(0, 4 * i) + P0(2, 1) * xyzw(0, 4 * i + 1) + P0(2, 2) * xyzw(0, 4 * i + 2) + P0(2, 3) * xyzw(0, 4 * i + 3);

        ASSERT_NEAR(xy0(0, 2 * i), x0 / z0, 1e-6);
        ASSERT_NEAR(xy0(0, 2 * i + 1), y0 / z0, 1e-6);

        double x1 = P1(0, 0) * xyzw(0, 4 * i) + P1(0, 1) * xyzw(0, 4 * i + 1) + P1(0, 2) * xyzw(0, 4 * i + 2) + P1(0, 3) * xyzw(0, 4 * i + 3);
        double y1 = P1(1, 0) * xyzw(0, 4 * i) + P1(1, 1) * xyzw(0, 4 * i + 1) + P1(1, 2) * xyzw(0, 4 * i + 2) + P1(1, 3) * xyzw(0, 4 * i + 3);
        double z1 = P1(2, 0) * xyzw(0, 4 * i) + P1(2, 1) * xyzw(0, 4 * i + 1) + P1(2, 2) * xyzw(0, 4 * i + 2) + P1(2, 3) * xyzw(0, 4 * i + 3);

        ASSERT_NEAR(xy1(0, 2 * i), x1 / z1, 1e-6);
        ASSERT_NEAR(xy1(0, 2 * i + 1), y1 / z1, 1e-6);
    }    
}

INSTANTIATE_TEST_CASE_P(StdSynthScenes,
                        DltTriangulationProjective,
                        testing::Values(new SphereSceneCreator(), new CubeSceneCreator()));


TEST(FindHomographyLinear, NoiselessSynthDataset) {
    RNG rng(0);
    int num_points = 1000; // 5 is the minimum acceptable value

    Mat_<double> H = Mat::zeros(4, 4, CV_64F);
    while (abs(determinant(H)) < 1e-3)
        rng.fill(H, RNG::UNIFORM, -1, 1);
    H /= pow(abs(determinant(H)), 0.25);

    Mat_<double> xyzw1(1, num_points * 4);
    Mat_<double> xyzw2(1, num_points * 4);

    rng.fill(xyzw1, RNG::UNIFORM, -1, 1);
 
    for (int i = 0; i < num_points; ++i) {
        xyzw2(0, 4 * i) = H(0, 0) * xyzw1(0, 4 * i) + H(0, 1) * xyzw1(0, 4 * i + 1) + H(0, 2) * xyzw1(0, 4 * i + 2) + H(0, 3) * xyzw1(0, 4 * i + 3);
        xyzw2(0, 4 * i + 1) = H(1, 0) * xyzw1(0, 4 * i) + H(1, 1) * xyzw1(0, 4 * i + 1) + H(1, 2) * xyzw1(0, 4 * i + 2) + H(1, 3) * xyzw1(0, 4 * i + 3);
        xyzw2(0, 4 * i + 2) = H(2, 0) * xyzw1(0, 4 * i) + H(2, 1) * xyzw1(0, 4 * i + 1) + H(2, 2) * xyzw1(0, 4 * i + 2) + H(2, 3) * xyzw1(0, 4 * i + 3);
        xyzw2(0, 4 * i + 3) = H(3, 0) * xyzw1(0, 4 * i) + H(3, 1) * xyzw1(0, 4 * i + 1) + H(3, 2) * xyzw1(0, 4 * i + 2) + H(3, 3) * xyzw1(0, 4 * i + 3);
    }

    Mat_<double> H_found = FindHomographyLinear(xyzw1, xyzw2);

    H /= H(3, 3);
    H_found /= H_found(3, 3);

    ASSERT_LT(norm(H, H_found, NORM_INF), 1e-6);
}


TEST(EigenDecompose, CanDecomposeRotationMat) {
    Mat_<double> mat(2, 2);
    mat(0, 0) = 0; mat(0, 1) = -1;
    mat(1, 0) = 1; mat(1, 1) = 0;

    Mat_<double> vals, vecs;
    EigenDecompose(mat, vals, vecs);

    complex<double> val1(vals(0, 0), vals(0, 1));
    complex<double> val2(vals(0, 2), vals(0, 3));
    ASSERT_NEAR(val1.real(), val2.real(), 1e-6);
    ASSERT_NEAR(val1.imag(), -val2.imag(), 1e-6);
    ASSERT_NEAR(0, val1.real(), 1e-6);
    ASSERT_NEAR(1, max(val1.imag(), val2.imag()), 1e-6);
}