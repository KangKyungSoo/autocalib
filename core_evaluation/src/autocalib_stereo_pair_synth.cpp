#pragma warning(disable: 4800)
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <core/include/core.h>
#include <evaluation/include/evaluation.h>

using namespace std;
using namespace cv;
using namespace autocalib;
using namespace autocalib::evaluation;

void ParseArgs(int argc, char **argv);

Ptr<PointCloudSceneCreator> scene_creator = new SphereSceneCreator();
int num_points = 1000;
int num_frames = 2;
Rect viewport = Rect(0, 0, 640, 480);
Mat_<double> K_gold;
int seed = 0; // No seed
double noise_stddev = -1; // No noise
bool create_images = false;

int main(int argc, char **argv) {
    try {
        ParseArgs(argc, argv);

        if (K_gold.empty()) {
            K_gold = Mat::eye(3, 3, CV_64F);
            K_gold(0, 0) = K_gold(1, 1) = viewport.width + viewport.height;
            K_gold(0, 2) = viewport.width * 0.5;
            K_gold(1, 2) = viewport.height * 0.5;
        }
        cout << "K_gold =\n" << K_gold << endl;

        RNG rng;
        if (seed > 0) {
            rng.state = seed;
            theRNG() = rng;
            srand(seed);
        }

        Mat_<double> rvec, R;
        Ptr<PointCloudScene> scene;

        // Generate synthetic scene

        scene = scene_creator->Create(num_points, rng);
        rvec = Mat::zeros(3, 1, CV_64F);
        rng.fill(rvec, RNG::UNIFORM, -1, 1);
        Rodrigues(rvec, R);
        scene->set_R(R);

        vector<RigidCamera> left_cameras(2);
        vector<RigidCamera> right_cameras(2);
        FeaturesCollection features_collection;
        MatchesCollection matches_collection;

        // Generate cameras and shots       

        Mat_<double> rel_T(3, 1);
        rel_T(0, 0) = 0.5; rel_T(1, 0) = rel_T(2, 0) = 0;

        Mat_<double> T(3, 1);

        detail::ImageFeatures features;

        for (int i = 0; i < num_frames; ++i) {
            while (true) {
                rng.fill(rvec, RNG::NORMAL, 0, 0.05);
                Rodrigues(rvec, R);

                rng.fill(T, RNG::NORMAL, 0, 1);
                T(0, 0) *= 2; T(2, 0) += -10;

                left_cameras[i] = RigidCamera::LocalToWorld(K_gold, R, -R * rel_T + T);
                right_cameras[i] = RigidCamera::LocalToWorld(K_gold, R, R * rel_T + T);

                scene->TakeShot(left_cameras[i], viewport, features);
                if (features.keypoints.size() < num_points / 3)
                    continue;

                scene->TakeShot(right_cameras[i], viewport, features);
                if (features.keypoints.size() < num_points / 3)
                    continue;

                break;
            }
        }

        for (int i = 0; i < num_frames; ++i) {
            Ptr<detail::ImageFeatures> left_features = new detail::ImageFeatures();
            scene->TakeShot(left_cameras[i], viewport, *left_features);
            features_collection[2 * i] = left_features;

            Ptr<detail::ImageFeatures> right_features = new detail::ImageFeatures();
            scene->TakeShot(right_cameras[i], viewport, *right_features);
            features_collection[2 * i + 1] = right_features;
        }

        // Add noise

        if (noise_stddev > 0) {
            cout << "\nAdding noise...\n";
            for (int i = 0; i < num_frames; ++i) {
                Mat_<float> noise_l(1, 2 * features_collection.find(2 * i)->second->keypoints.size());
                Mat_<float> noise_r(1, 2 * features_collection.find(2 * i + 1)->second->keypoints.size());

                // Final noise RMS is determined by sqrt(noise_x^2 + noise_y^2),
                // so we split by sqrt(2) to get desired noise
                rng.fill(noise_l, RNG::NORMAL, 0, noise_stddev / sqrt(2.));
                rng.fill(noise_r, RNG::NORMAL, 0, noise_stddev / sqrt(2.));

                double total_noise = 0;
                Ptr<detail::ImageFeatures> features = features_collection.find(2 * i)->second;

                for (size_t j = 0; j < features->keypoints.size(); ++j) {
                    features->keypoints[j].pt.x += noise_l(0, 2 * j);
                    features->keypoints[j].pt.y += noise_l(0, 2 * j + 1);
                    total_noise += Sqr(noise_l(0, 2 * j)) + Sqr(noise_l(0, 2 * j + 1));
                }
                cout << "Pair #" << i << " left frame RMS error = " << sqrt(total_noise / features->keypoints.size()) << endl;

                total_noise = 0;
                features = features_collection.find(2 * i + 1)->second;
                for (size_t j = 0; j < features->keypoints.size(); ++j) {
                    features->keypoints[j].pt.x += noise_r(0, 2 * j);
                    features->keypoints[j].pt.y += noise_r(0, 2 * j + 1);
                    total_noise += Sqr(noise_r(0, 2 * j)) + Sqr(noise_r(0, 2 * j + 1));
                }
                cout << "Pair #" << i << " right frame RMS error = " << sqrt(total_noise / features->keypoints.size()) << endl;
            }
        }

        // Save images if it's needed

        if (create_images) {
            for (int i = 0; i < num_frames; ++i) {
                Mat img_l = CreateImage(*(features_collection.find(2 * i)->second));
                stringstream name; name << "left_camera" << i << ".jpg";
                imwrite(name.str(), img_l);

                Mat img_r = CreateImage(*(features_collection.find(2 * i + 1)->second));
                name.str(""); name << "right_camera" << i << ".jpg";
                imwrite(name.str(), img_r);
            }
        }

        // Find fundmental matrix

        cout << "\nFinding F between #0 pair images...";

        Ptr<vector<DMatch> > matches_lr0 = new vector<DMatch>();
        MatchSyntheticShots(*(features_collection.find(0)->second),
                            *(features_collection.find(1)->second),
                            *matches_lr0);
        matches_collection[make_pair(0, 1)] = matches_lr0;

        cout << " #matches = " << matches_lr0->size();

        Mat_<double> xy_l0, xy_r0;
        ExtractMatchedKeypoints(*(features_collection.find(0)->second),
                                *(features_collection.find(1)->second),
                                *matches_lr0, xy_l0, xy_r0);

        vector<uchar> inlier_mask0;
        Mat_<double> F0 = findFundamentalMat(Mat(xy_l0).reshape(2), Mat(xy_r0).reshape(2), inlier_mask0, RANSAC, 0.1);

        int num_inliers0 = 0;
        for (size_t i = 0; i < inlier_mask0.size(); ++i)
            if (inlier_mask0[i])
                num_inliers0++;                

        cout << ", #inliers = " << num_inliers0
             << ", p2l dist RMS = " << CalcRmsEpipolarDistance(xy_r0, xy_l0, F0) << endl;

        cout << "Finding F between #1 pair images...";

        Ptr<vector<DMatch> > matches_lr1 = new vector<DMatch>();
        MatchSyntheticShots(*(features_collection.find(2)->second),
                            *(features_collection.find(3)->second),
                            *matches_lr1);
        matches_collection[make_pair(2, 3)] = matches_lr1;

        cout << " #matches = " << matches_lr1->size();

        Mat_<double> xy_l1, xy_r1;
        ExtractMatchedKeypoints(*(features_collection.find(2)->second),
                                *(features_collection.find(3)->second),
                                *matches_lr1, xy_l1, xy_r1);

        vector<uchar> inlier_mask1;
        Mat_<double> F1 = findFundamentalMat(Mat(xy_l1).reshape(2), Mat(xy_r1).reshape(2), inlier_mask1, RANSAC, 0.1);

        int num_inliers1 = 0;
        for (size_t i = 0; i < inlier_mask1.size(); ++i)
            if (inlier_mask1[i])
                num_inliers1++;

        cout << ", #inliers = " << num_inliers1
             << ", p2l dist RMS = " << CalcRmsEpipolarDistance(xy_r1, xy_l1, F1) << endl;

        // Extract camera matrices

        Mat_<double> P_l0 = Mat::eye(3, 4, CV_64F);
        Mat_<double> P_r0 = Extract2ndCameraMatFromF(F0);

        // Find structure

        DltTriangulation dlt;

        Mat_<double> xyzw0;
        dlt.triangulate(ProjectiveCamera(P_l0), ProjectiveCamera(P_r0), xy_l0, xy_r0, xyzw0);

        Mat_<double> xyzw1;
        dlt.triangulate(ProjectiveCamera(P_l0), ProjectiveCamera(P_r0), xy_l1, xy_r1, xyzw1);

        cout << "\n(F0) DLT reprojection RMS errors (l0 r0 l1 r1) = ("
             << CalcRmsReprojectionError(xy_l0, P_l0, xyzw0) << " "
             << CalcRmsReprojectionError(xy_r0, P_r0, xyzw0) << " "
             << CalcRmsReprojectionError(xy_l1, P_l0, xyzw1) << " "
             << CalcRmsReprojectionError(xy_r1, P_r0, xyzw1) << ")\n";

        // Check if we can find structure using F1 instead of F0

        Mat_<double> P_r_ = Extract2ndCameraMatFromF(F1);

        Mat_<double> xyzw0_;
        dlt.triangulate(ProjectiveCamera(P_l0), ProjectiveCamera(P_r_), xy_l0, xy_r0, xyzw0_);

        Mat_<double> xyzw1_;
        dlt.triangulate(ProjectiveCamera(P_l0), ProjectiveCamera(P_r_), xy_l1, xy_r1, xyzw1_);

        cout << "(F1) DLT reprojection RMS errors (l0 r0 l1 r1) = ("
             << CalcRmsReprojectionError(xy_l0, P_l0, xyzw0_) << " "
             << CalcRmsReprojectionError(xy_r0, P_r_, xyzw0_) << " "
             << CalcRmsReprojectionError(xy_l1, P_l0, xyzw1_) << " "
             << CalcRmsReprojectionError(xy_r1, P_r_, xyzw1_) << ")\n";

        // Match two stereo pairs

        cout << "\nMatching two stereo pairs using left images...";

        Ptr<vector<DMatch> > matches_ll = new vector<DMatch>();
        MatchSyntheticShots(*(features_collection.find(0)->second),
                            *(features_collection.find(2)->second),
                            *matches_ll);
        matches_collection[make_pair(0, 2)] = matches_ll;

        cout << " #matches = " << matches_ll->size() << endl;

        // Leave only common part of point clouds

        vector<pair<int, int> > lr0_lr1_indices;
        Intersect(*matches_lr0, *matches_lr1, *matches_ll, lr0_lr1_indices);

        Mat_<double> xy_l0_buf(1, lr0_lr1_indices.size() * 2);
        Mat_<double> xy_r0_buf(1, lr0_lr1_indices.size() * 2);
        Mat_<double> xy_l1_buf(1, lr0_lr1_indices.size() * 2);
        Mat_<double> xy_r1_buf(1, lr0_lr1_indices.size() * 2);
        Mat_<double> xyzw0_buf(1, lr0_lr1_indices.size() * 4);
        Mat_<double> xyzw1_buf(1, lr0_lr1_indices.size() * 4);

        for (size_t i = 0; i < lr0_lr1_indices.size(); ++i) {
            int i0 = lr0_lr1_indices[i].first;
            int i1 = lr0_lr1_indices[i].second;

            xy_l0_buf(0, 2 * i) = xy_l0(0, 2 * i0);
            xy_l0_buf(0, 2 * i + 1) = xy_l0(0, 2 * i0 + 1);

            xy_r0_buf(0, 2 * i) = xy_r0(0, 2 * i0);
            xy_r0_buf(0, 2 * i + 1) = xy_r0(0, 2 * i0 + 1);

            xy_l1_buf(0, 2 * i) = xy_l1(0, 2 * i1);
            xy_l1_buf(0, 2 * i + 1) = xy_l1(0, 2 * i1 + 1);

            xy_r1_buf(0, 2 * i) = xy_r1(0, 2 * i1);
            xy_r1_buf(0, 2 * i + 1) = xy_r1(0, 2 * i1 + 1);

            xyzw0_buf(0, 4 * i) = xyzw0(0, 4 * i0);
            xyzw0_buf(0, 4 * i + 1) = xyzw0(0, 4 * i0 + 1);
            xyzw0_buf(0, 4 * i + 2) = xyzw0(0, 4 * i0 + 2);
            xyzw0_buf(0, 4 * i + 3) = xyzw0(0, 4 * i0 + 3);

            xyzw1_buf(0, 4 * i) = xyzw1(0, 4 * i1);
            xyzw1_buf(0, 4 * i + 1) = xyzw1(0, 4 * i1 + 1);
            xyzw1_buf(0, 4 * i + 2) = xyzw1(0, 4 * i1 + 2);
            xyzw1_buf(0, 4 * i + 3) = xyzw1(0, 4 * i1 + 3);
        }

        xy_l0 = xy_l0_buf;
        xy_r0 = xy_r0_buf;
        xy_l1 = xy_l1_buf;
        xy_r1 = xy_r1_buf;
        xyzw0 = xyzw0_buf;
        xyzw1 = xyzw1_buf;

        // Find homography mapping the 1st cloud to the 2nd one

        int num_points_common = xyzw0.cols / 4;

        cout << "\nFinding H01 using " << num_points_common << " common points (point)...\n";

        Mat_<double> H01 = FindHomographyLinear(xyzw0, xyzw1);

        Mat_<double> xyzw1_mapped(xyzw0.size(), xyzw0.type());
        for (int i = 0; i < num_points_common; ++i) {
            xyzw1_mapped(0, 4 * i) = H01(0, 0) * xyzw0(0, 4 * i) + H01(0, 1) * xyzw0(0, 4 * i + 1) + H01(0, 2) * xyzw0(0, 4 * i + 2) + H01(0, 3) * xyzw0(0, 4 * i + 3);
            xyzw1_mapped(0, 4 * i + 1) = H01(1, 0) * xyzw0(0, 4 * i) + H01(1, 1) * xyzw0(0, 4 * i + 1) + H01(1, 2) * xyzw0(0, 4 * i + 2) + H01(1, 3) * xyzw0(0, 4 * i + 3);
            xyzw1_mapped(0, 4 * i + 2) = H01(2, 0) * xyzw0(0, 4 * i) + H01(2, 1) * xyzw0(0, 4 * i + 1) + H01(2, 2) * xyzw0(0, 4 * i + 2) + H01(2, 3) * xyzw0(0, 4 * i + 3);
            xyzw1_mapped(0, 4 * i + 3) = H01(3, 0) * xyzw0(0, 4 * i) + H01(3, 1) * xyzw0(0, 4 * i + 1) + H01(3, 2) * xyzw0(0, 4 * i + 2) + H01(3, 3) * xyzw0(0, 4 * i + 3);
        }

        cout << "Reprojection RMS error after mapping (l1 r1) = ("
             << CalcRmsReprojectionError(xy_l1, P_l0, xyzw1_mapped) << " "
             << CalcRmsReprojectionError(xy_r1, P_r0, xyzw1_mapped) << ")\n";

        // Finding plane-at-infinity

        cout << "\nFinding plane-at-infinity...\n";

        Mat evals, evecs;
        EigenDecompose(H01.t(), evals, evecs);
        cout << "Eigenvalues of H01.t() = " << evals << endl;

        Mat_<double> p_inf = CalcPlaneAtInfinity(H01);
        p_inf /= p_inf(3, 0);
        cout << "Plane-at-infinity = " << p_inf << endl;

        // Affine rectification

        cout << "\nAffine rectification...\n";

        Mat_<double> Hpa = Mat::eye(4, 4, CV_64F);
        Hpa(3, 0) = -p_inf(0, 0); Hpa(3, 1) = -p_inf(1, 0); Hpa(3, 2) = -p_inf(2, 0);

        H01 = Hpa.inv() * H01 * Hpa;

        P_l0 = P_l0 * Hpa;
        P_r0 = P_r0 * Hpa;

        xyzw0 = Hpa.inv() * xyzw0.reshape(num_points_common).t();
        xyzw1 = Hpa.inv() * xyzw1.reshape(num_points_common).t();
        xyzw0 = Mat(xyzw0.t()).reshape(0, 1);
        xyzw1 = Mat(xyzw1.t()).reshape(0, 1);

        cout << "Reprojection RMS error after affine rectification (l0 r0 l1 r1) = ("
             << CalcRmsReprojectionError(xy_l0, P_l0, xyzw0) << " "
             << CalcRmsReprojectionError(xy_r0, P_r0, xyzw0) << " "
             << CalcRmsReprojectionError(xy_l1, P_l0, xyzw1) << " "
             << CalcRmsReprojectionError(xy_r1, P_r0, xyzw1) << ")\n";

        // Linear calibration

        cout << "\nLinear calibrating...\n";

        HomographiesP2 Hs_inf;

        // Stereo pair relative rotation can be very close to the identity matrix. That
        // can lead to numerical instability in K estimation process, so we avoid using those
        // rotations in the linear autocalibration algorithm.

        //Hs_inf[make_pair(0, 1)] = P_r0(Rect(0, 0, 3, 3));
        Hs_inf[make_pair(0, 2)] = Mat(P_l0 * H01.inv())(Rect(0, 0, 3, 3));

        Mat_<double> K_linear = CalibRotationalCameraLinearNoSkew(Hs_inf);
        cout << "K_linear = \n" << K_linear << endl;

        // Metric rectification

        cout << "\nMetric rectification...\n";

        Mat_<double> Ham = Mat::eye(4, 4, CV_64F);
        Mat Ham_3x3 = Ham(Rect(0, 0, 3, 3));
        K_linear.copyTo(Ham_3x3);

        H01 = Ham.inv() * H01 * Ham;
        H01 /= H01(3, 3);

        cout << "Metric H01 = \n" << H01 << endl;

        Mat_<double> R01 = H01(Rect(0, 0, 3, 3));
        Mat_<double> T01 = H01(Rect(3, 0, 1, 3));

        P_l0 = P_l0 * Ham;
        P_r0 = P_r0 * Ham;

        xyzw0 = Ham.inv() * xyzw0.reshape(num_points_common).t();
        xyzw1 = Ham.inv() * xyzw1.reshape(num_points_common).t();
        xyzw0 = Mat(xyzw0.t()).reshape(0, 1);
        xyzw1 = Mat(xyzw1.t()).reshape(0, 1);

        cout << "Reprojection RMS error after metric rectification (l0 r0 l1 r1) = ("
             << CalcRmsReprojectionError(xy_l0, P_l0, xyzw0) << " "
             << CalcRmsReprojectionError(xy_r0, P_r0, xyzw0) << " "
             << CalcRmsReprojectionError(xy_l1, P_l0, xyzw1) << " "
             << CalcRmsReprojectionError(xy_r1, P_r0, xyzw1) << ")\n";

        Mat_<double> F01 = K_linear.inv().t() * CrossProductMat(T01) * R01 * K_linear.inv();
        cout << "Point-to-line distance RMS = " << CalcRmsEpipolarDistance(xy_l1, xy_l0, F01) << endl;

        // Refine reconstruction

        cout << "\nRefining metric reconstruction...\n";

        AbsoluteMotions motions;
        motions[0] = Motion(Mat::eye(3, 3, CV_64F), Mat::zeros(3, 1, CV_64F));
        motions[1] = Motion(R01, T01);

        RigidCamera P_r0_m = RigidCamera::FromProjectiveMat(P_r0);
        RefineStereoCamera(P_r0_m, motions, features_collection, matches_collection);

        Mat_<double> K_refined = P_r0_m.K();
        cout << "K_refined = \n" << K_refined << endl;
    }
    catch (const exception &e) {
        cout << "Error: " << e.what() << endl;
    }
    return 0;
}

void ParseArgs(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--scene") {
            if (string(argv[i + 1]) == "sphere")
                scene_creator = new SphereSceneCreator();
            else if (string(argv[i + 1]) == "cube")
                scene_creator = new CubeSceneCreator();
            else
                throw runtime_error(string("Unknown synthetic scene type: ") + argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--num-points")
            num_points = atoi(argv[++i]);
        else if (string(argv[i]) == "--viewport") {
            viewport = Rect(atoi(argv[i + 1]), atoi(argv[i + 2]),
                            atoi(argv[i + 3]), atoi(argv[i + 4]));
            i += 4;
        }
        else if (string(argv[i]) == "--K-gold") {
            K_gold = Mat::eye(3, 3, CV_64F);
            K_gold(0, 0) = atof(argv[i + 1]);
            K_gold(0, 1) = atof(argv[i + 2]);
            K_gold(0, 2) = atof(argv[i + 3]);
            K_gold(1, 1) = atof(argv[i + 4]);
            K_gold(1, 2) = atof(argv[i + 5]);
            i += 5;
        }
        else if (string(argv[i]) == "--seed")
            seed = atoi(argv[++i]);
        else if (string(argv[i]) == "--noise-stddev")
            noise_stddev = atof(argv[++i]);
        else if (string(argv[i]) == "--create-images")
            create_images = (bool)atoi(argv[++i]);
        else
            throw runtime_error(string("Can't parse command line arg: ") + argv[i]);
    }
}

