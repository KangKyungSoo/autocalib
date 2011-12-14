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


TEST(CrossProductMat, CanRepresentCrossProduct) {
    RNG rng(0);
    Mat vec1(3, 1, CV_64F), vec2(3, 1, CV_64F);
    rng.fill(vec1, RNG::UNIFORM, -1, 1);
    rng.fill(vec2, RNG::UNIFORM, -1, 1);

    ASSERT_LT(norm(CrossProductMat(vec1) * vec2, vec1.cross(vec2), NORM_INF), 1e-6);
}


TEST(CameraMatFromFundamentalMat, CanRun) {
    Mat_<double> F = Mat::eye(3, 3, CV_64F);
    F(2, 2) = 0;

    ASSERT_NO_THROW(Mat P = CameraMatFromFundamentalMat(F));
}


class TriangulationMetric : public testing::TestWithParam<tr1::tuple<Ptr<ITringulationMethodCreator>,
                                                                     Ptr<IPointCloudSceneCreator> > > { };

TEST_P(TriangulationMetric, CanTriangulate) {
    RNG rng(0);
    int num_points = 1000;
    Rect viewport = Rect(0, 0, 640, 480);

    Ptr<ITriangulationMethod> method = tr1::get<0>(GetParam())->Create();
    Ptr<PointCloudScene> scene = tr1::get<1>(GetParam())->Create(num_points, rng);

    Mat_<double> K = Mat::eye(3, 3, CV_64F);
    K(0, 0) = K(1, 1) = viewport.width + viewport.height;
    K(0, 2) = viewport.width * 0.5;
    K(1, 2) = viewport.height * 0.5;

    vector<RigidCamera> cameras(2);
    FeaturesCollection features_collection;

    for (int i = 0; i < 2; ++i) {
        Mat_<double> center = Mat::zeros(3, 1, CV_64F);
        center(0, 0) = i * 2 - 1; center(2, 0) = -10;
        Mat_<double> rvec = Mat::zeros(1, 3, CV_64F);
        rvec(0, 0) = 0.1; rvec(0, 1) = 0.1; rvec(0, 2) = 0.1;
        Mat R; Rodrigues(rvec * (i * 2 - 1), R);
        cameras[i] = RigidCamera::FromLocalToWorld(K, R, center);
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

    Mat_<double> xyzw;
    method->triangulate(cameras[0], cameras[1], xy0, xy1, xyzw);

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
                        TriangulationMetric,
                        testing::Combine(
                            testing::Values(new DltTriangulationCreator(), new IterativeTriangulationCreator()),
                            testing::Values(new SphereSceneCreator(), new CubeSceneCreator())));

// Collect information about the RMS error as a function of the distance and the baseline
/*

class TriangulationMetric : public testing::TestWithParam<tr1::tuple<Ptr<ITringulationMethodCreator>,
                                                                     Ptr<IPointCloudSceneCreator> > > { };

TEST_P(TriangulationMetric, CanTriangulate) {
    RNG rng(0);
    int num_points = 1000;
    Rect viewport = Rect(0, 0, 640, 480);

    Ptr<ITriangulationMethod> method = tr1::get<0>(GetParam())->Create();
    Ptr<PointCloudScene> scene = tr1::get<1>(GetParam())->Create(num_points, rng);

    Mat_<double> K = Mat::eye(3, 3, CV_64F);
    K(0, 0) = K(1, 1) = viewport.width + viewport.height;
    K(0, 2) = viewport.width * 0.5;
    K(1, 2) = viewport.height * 0.5;

    vector<RigidCamera> cameras(2);
    FeaturesCollection features_collection;

    ofstream f("rms_err_of_dist.csv");

    for (double dist = -1; dist > -50; dist -= 1.5) {
        for (double baseline = 0.01; baseline < 5; baseline += 0.03) {
            for (int i = 0; i < 2; ++i) {
                Mat_<double> center = Mat::zeros(3, 1, CV_64F);
                center(0, 0) = (i * 2 - 1) * baseline; center(2, 0) = dist;//-10;
                Mat_<double> rvec = Mat::zeros(1, 3, CV_64F);
                rvec(0, 0) = 0.1; rvec(0, 1) = 0.1; rvec(0, 2) = 0.1;
                Mat R; Rodrigues(rvec * (i * 2 - 1), R);
                cameras[i] = RigidCamera::FromLocalToWorld(K, R, center);
                Ptr<detail::ImageFeatures> features = new detail::ImageFeatures();
                scene->TakeShot(cameras[i], viewport, *features);
                features_collection[i] = features;
            }

            vector<DMatch> matches;
            MatchSyntheticShots(*(features_collection.find(0)->second),
                                *(features_collection.find(1)->second),
                                matches);

            if (matches.size() == 0) {
                f << dist << " " << baseline * 2 << " NaN\n";
                continue;
            }

            RNG rng(0);
            Mat_<double> xy0(1, matches.size() * 2);
            Mat_<double> xy1(1, matches.size() * 2);
            const detail::ImageFeatures &f0 = *(features_collection.find(0)->second);
            const detail::ImageFeatures &f1 = *(features_collection.find(1)->second);
            for (size_t i = 0; i < matches.size(); ++i)  {
                xy0(0, 2 * i) = f0.keypoints[matches[i].queryIdx].pt.x + rng.gaussian(0.5);
                xy0(0, 2 * i + 1) = f0.keypoints[matches[i].queryIdx].pt.y + rng.gaussian(0.5);
                xy1(0, 2 * i) = f1.keypoints[matches[i].trainIdx].pt.x + rng.gaussian(0.5);
                xy1(0, 2 * i + 1) = f1.keypoints[matches[i].trainIdx].pt.y + rng.gaussian(0.5);
            }

            Mat_<double> xyzw;
            method->triangulate(cameras[0], cameras[1], xy0, xy1, xyzw);

            double total_err = 0;

            for (size_t i = 0; i < matches.size(); ++i) {
                int point_idx = f0.descriptors.at<int>(matches[i].queryIdx);
                Point3d point_gold = scene->localPointAt(point_idx);
                Point3d point_found(xyzw(0, 4 * i) / xyzw(0, 4 * i + 3),
                                    xyzw(0, 4 * i + 1) / xyzw(0, 4 * i + 3),
                                    xyzw(0, 4 * i + 2) / xyzw(0, 4 * i + 3));
                double err = sqr(point_gold.x - point_found.x) +
                             sqr(point_gold.y - point_found.y) +
                             sqr(point_gold.z - point_found.z);
                total_err += err;

    //            ASSERT_NEAR(point_gold.x, point_found.x, 1e-5);
    //            ASSERT_NEAR(point_gold.y, point_found.y, 1e-5);
    //            ASSERT_NEAR(point_gold.z, point_found.z, 1e-5);
            }

            f << dist << " " << baseline * 2 << " " << sqrt(total_err / matches.size()) << endl;
        }
    }
}

INSTANTIATE_TEST_CASE_P(StdSynthScenes,
                        TriangulationMetric,
                        testing::Combine(
                            testing::Values(Ptr<ITringulationMethodCreator>(new DltTriangulationCreator())),
                            testing::Values(Ptr<IPointCloudSceneCreator>(new SphereSceneCreator()))));
*/


class TriangulationProjective : public testing::TestWithParam<tr1::tuple<Ptr<ITringulationMethodCreator>,
                                                                         Ptr<IPointCloudSceneCreator> > > { };

TEST_P(TriangulationProjective, CanTriangulate) {
    RNG rng(0);
    int num_points = 1000;
    Rect viewport = Rect(0, 0, 640, 480);    

    Ptr<ITriangulationMethod> method = tr1::get<0>(GetParam())->Create();
    Ptr<PointCloudScene> scene = tr1::get<1>(GetParam())->Create(num_points, rng);

    Mat_<double> K = Mat::eye(3, 3, CV_64F);
    K(0, 0) = K(1, 1) = viewport.width + viewport.height;
    K(0, 2) = viewport.width * 0.5;
    K(1, 2) = viewport.height * 0.5;

    vector<RigidCamera> cameras(2);
    FeaturesCollection features_collection;

    for (int i = 0; i < 2; ++i) {
        Mat_<double> center = Mat::zeros(3, 1, CV_64F);
        center(0, 0) = i * 2 - 1; center(2, 0) = -10;
        Mat_<double> rvec = Mat::zeros(1, 3, CV_64F);
        rvec(0, 0) = 0.1; rvec(0, 1) = 0.1; rvec(0, 2) = 0.1;
        Mat R; Rodrigues(rvec * (i * 2 - 1), R);
        cameras[i] = RigidCamera::FromLocalToWorld(K, R, center);
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

    Mat_<double> P0 = cameras[0].P() * H_inv;
    Mat_<double> P1 = cameras[1].P() * H_inv;

    Mat_<double> xyzw;
    method->triangulate(ProjectiveCamera(P0), ProjectiveCamera(P1), xy0, xy1, xyzw);

    for (size_t i = 0; i < matches.size(); ++i) {
        double x0 = P0(0, 0) * xyzw(0, 4 * i) + P0(0, 1) * xyzw(0, 4 * i + 1) + P0(0, 2) * xyzw(0, 4 * i + 2) + P0(0, 3) * xyzw(0, 4 * i + 3);
        double y0 = P0(1, 0) * xyzw(0, 4 * i) + P0(1, 1) * xyzw(0, 4 * i + 1) + P0(1, 2) * xyzw(0, 4 * i + 2) + P0(1, 3) * xyzw(0, 4 * i + 3);
        double z0 = P0(2, 0) * xyzw(0, 4 * i) + P0(2, 1) * xyzw(0, 4 * i + 1) + P0(2, 2) * xyzw(0, 4 * i + 2) + P0(2, 3) * xyzw(0, 4 * i + 3);

        ASSERT_NEAR(xy0(0, 2 * i), x0 / z0, 1e-4);
        ASSERT_NEAR(xy0(0, 2 * i + 1), y0 / z0, 1e-4);

        double x1 = P1(0, 0) * xyzw(0, 4 * i) + P1(0, 1) * xyzw(0, 4 * i + 1) + P1(0, 2) * xyzw(0, 4 * i + 2) + P1(0, 3) * xyzw(0, 4 * i + 3);
        double y1 = P1(1, 0) * xyzw(0, 4 * i) + P1(1, 1) * xyzw(0, 4 * i + 1) + P1(1, 2) * xyzw(0, 4 * i + 2) + P1(1, 3) * xyzw(0, 4 * i + 3);
        double z1 = P1(2, 0) * xyzw(0, 4 * i) + P1(2, 1) * xyzw(0, 4 * i + 1) + P1(2, 2) * xyzw(0, 4 * i + 2) + P1(2, 3) * xyzw(0, 4 * i + 3);

        ASSERT_NEAR(xy1(0, 2 * i), x1 / z1, 1e-4);
        ASSERT_NEAR(xy1(0, 2 * i + 1), y1 / z1, 1e-4);
    }    
}

INSTANTIATE_TEST_CASE_P(StdSynthScenes,
                        TriangulationProjective,
                        testing::Combine(
                            testing::Values(new DltTriangulationCreator(), new IterativeTriangulationCreator()),
                            testing::Values(new SphereSceneCreator(), new CubeSceneCreator())));


TEST(FindHomographyP3Linear, NoiselessSynthDataset) {
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

    Mat_<double> H_found = FindHomographyP3Linear(xyzw1, xyzw2);

    H /= H(3, 3);
    H_found /= H_found(3, 3);

    ASSERT_LT(norm(H, H_found, NORM_INF), 1e-6);
}


TEST(FindHomographyP3Linear, CanFindEuclideanMap) {
    int num_points = 1000;

    Mat_<double> xyzw1(1, num_points * 4);
    Mat_<double> xyzw2(1, num_points * 4);

    RNG rng(0);
    rng.fill(xyzw1, RNG::UNIFORM, -1, 1);

    Mat_<double> rvec(3, 1);
    rvec(0, 0) = 0.1; rvec(1, 0) = 0.1; rvec(2, 0) = 0.1;
    Mat_<double> tvec(3, 1);
    tvec(0, 0) = 0; tvec(1, 0) = 0; tvec(2, 0) = 10;
    Mat_<double> H = Mat::eye(4, 4, CV_64F);
    Mat tmp;
    tmp = H(Rect(0, 0, 3, 3));
    Mat R;
    Rodrigues(rvec, R);
    R.copyTo(tmp);
    tmp = H(Rect(3, 0, 1, 3));
    tvec.copyTo(tmp);
 
    for (int i = 0; i < num_points; ++i) {
        xyzw2(0, 4 * i) = H(0, 0) * xyzw1(0, 4 * i) + H(0, 1) * xyzw1(0, 4 * i + 1) + H(0, 2) * xyzw1(0, 4 * i + 2) + H(0, 3) * xyzw1(0, 4 * i + 3);
        xyzw2(0, 4 * i + 1) = H(1, 0) * xyzw1(0, 4 * i) + H(1, 1) * xyzw1(0, 4 * i + 1) + H(1, 2) * xyzw1(0, 4 * i + 2) + H(1, 3) * xyzw1(0, 4 * i + 3);
        xyzw2(0, 4 * i + 2) = H(2, 0) * xyzw1(0, 4 * i) + H(2, 1) * xyzw1(0, 4 * i + 1) + H(2, 2) * xyzw1(0, 4 * i + 2) + H(2, 3) * xyzw1(0, 4 * i + 3);
        xyzw2(0, 4 * i + 3) = H(3, 0) * xyzw1(0, 4 * i) + H(3, 1) * xyzw1(0, 4 * i + 1) + H(3, 2) * xyzw1(0, 4 * i + 2) + H(3, 3) * xyzw1(0, 4 * i + 3);
    }

    Mat_<double> H_found = FindHomographyP3Linear(xyzw1, xyzw2);
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


TEST(CalcPlaneAtInfinity, CanCalcFromEuclideanTransformation) {
    Mat_<double> H = Mat::eye(4, 4, CV_64F);
    Mat_<double> rvec(3, 1);
    rvec(0, 0) = 0.1; rvec(1, 0) = 0.1; rvec(2, 0) = 0.1;
    Mat R;
    Rodrigues(rvec, R);
    Mat tmp = H(Rect(0, 0, 3, 3));
    R.copyTo(tmp);
    H(2, 3) = 1;

    Mat_<double> pinf = CalcPlaneAtInfinity(H);
    ASSERT_NEAR(0, pinf(0, 0), 1e-6);
    ASSERT_NEAR(0, pinf(1, 0), 1e-6);
    ASSERT_NEAR(0, pinf(2, 0), 1e-6);
    ASSERT_NEAR(1, pinf(3, 0), 1e-6);
}


TEST(CameraCentre, CanFindForRandomMat3x4) {
    RNG rng(0);
    Mat_<double> P = Mat::zeros(3, 4, CV_64F);
    rng.fill(P, RNG::UNIFORM, -1, 1);
    ASSERT_NEAR(0, norm(P * CameraCentre(P)), 1e-6);
}


TEST(PseudoInverse, CanFindForRandomMat3x4) {
    RNG rng(0);
    Mat_<double> P = Mat::zeros(3, 3, CV_64F);
    rng.fill(P, RNG::UNIFORM, -1, 1);
    Mat P_inv = PseudoInverse(P);
    ASSERT_NEAR(0, norm(P * P_inv * P, P), 1e-6);
    ASSERT_NEAR(0, norm(P_inv * P * P_inv, P_inv), 1e-6);
}
