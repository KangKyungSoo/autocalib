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
        if (seed > 0)
            rng.state = seed;

        Mat_<double> rvec, R;
        Ptr<PointCloudScene> scene;

        // Generate synthetic scene

        scene = scene_creator->Create(num_points, rng);
        rvec = Mat::zeros(3, 1, CV_64F);
        rng.fill(rvec, RNG::UNIFORM, -1, 1);
        Rodrigues(rvec, R);
        scene->set_R(R);

        vector<RigidCamera> left_cameras;
        vector<RigidCamera> right_cameras;
        FeaturesCollection right_features_collection;
        FeaturesCollection left_features_collection;

        // Generate cameras and shots

        Mat_<double> offset(3, 1);
        offset(0, 0) = 1; offset(1, 0) = offset(2, 0) = 0;

        rvec = Mat::zeros(3, 1, CV_64F);
        rvec(0, 0) = rvec(2, 0) = CV_PI / 40; rvec(1, 0) = CV_PI / 20;
        Rodrigues(rvec, R);

        Mat_<double> T(3, 1);
        T(0, 0) = -2; T(1, 0) = 0; T(2, 0) = -10;

        left_cameras.push_back(RigidCamera::LocalToWorld(K_gold, R, -R * offset + T));
        right_cameras.push_back(RigidCamera::LocalToWorld(K_gold, R, R * offset + T));

        T(0, 0) = 2; T(1, 0) = 0; T(2, 0) = -10;

        left_cameras.push_back(RigidCamera::LocalToWorld(K_gold, R.t(), -R.t() * offset + T));
        right_cameras.push_back(RigidCamera::LocalToWorld(K_gold, R.t(), R.t() * offset + T));

        for (int i = 0; i < num_frames; ++i) {
            Ptr<detail::ImageFeatures> left_features = new detail::ImageFeatures();
            scene->TakeShot(left_cameras[i], viewport, *left_features);
            left_features_collection[i] = left_features;

            Ptr<detail::ImageFeatures> right_features = new detail::ImageFeatures();
            scene->TakeShot(right_cameras[i], viewport, *right_features);
            right_features_collection[i] = right_features;
        }

        // Save images if it's needed

        if (create_images) {
            for (int i = 0; i < num_frames; ++i) {
                Mat img_l, img_r;

                CreateImage(*(left_features_collection.find(i)->second), img_l);
                stringstream name; name << "left_camera" << i << ".jpg";
                imwrite(name.str(), img_l);

                CreateImage(*(right_features_collection.find(i)->second), img_r);
                name.str(""); name << "right_camera" << i << ".jpg";
                imwrite(name.str(), img_r);
            }
        }

        // Find fundmental matrix

        cout << "\nFinding F between #0 pair images...";

        vector<DMatch> matches_lr_0;
        MatchSyntheticShots(*(left_features_collection.find(0)->second),
                            *(right_features_collection.find(0)->second),
                            matches_lr_0);

        cout << " #matches = " << matches_lr_0.size();

        Mat xy_l_0, xy_r_0;
        ExtractMatchedKeypoints(*(left_features_collection.find(0)->second),
                                *(right_features_collection.find(0)->second),
                                matches_lr_0, xy_l_0, xy_r_0);

        vector<uchar> inlier_mask0;
        Mat_<double> F0 = findFundamentalMat(xy_l_0.reshape(2), xy_r_0.reshape(2), inlier_mask0, RANSAC, 0.1);

        int num_inliers0 = 0;
        for (size_t i = 0; i < inlier_mask0.size(); ++i)
            if (inlier_mask0[i])
                num_inliers0++;

        cout << ", #inliers = " << num_inliers0 << endl
             << "F0 = \n" << F0 << endl;

        cout << "Finding F between #1 pair images...";

        vector<DMatch> matches_lr_1;
        MatchSyntheticShots(*(left_features_collection.find(1)->second),
                            *(right_features_collection.find(1)->second),
                            matches_lr_1);

        cout << " #matches = " << matches_lr_1.size();

        Mat xy_l_1, xy_r_1;
        ExtractMatchedKeypoints(*(left_features_collection.find(1)->second),
                                *(right_features_collection.find(1)->second),
                                matches_lr_1, xy_l_1, xy_r_1);

        vector<uchar> inlier_mask1;
        Mat_<double> F1 = findFundamentalMat(xy_l_1.reshape(2), xy_r_1.reshape(2), inlier_mask1, RANSAC, 0.1);

        int num_inliers1 = 0;
        for (size_t i = 0; i < inlier_mask1.size(); ++i)
            if (inlier_mask1[i])
                num_inliers1++;

        cout << ", #inliers = " << num_inliers1 << endl
             << "F1 = \n" << F1 << endl;

        // Extract camera matrices

        Mat_<double> P_l = Mat::eye(3, 4, CV_64F);
        Mat_<double> P_r = Extract2ndCameraMatFromF(F0);

        // Find structure

        DltTriangulation dlt;

        Mat_<double> xyzw0;
        dlt.triangulate(ProjectiveCamera(P_l), ProjectiveCamera(P_r), xy_l_0, xy_r_0, xyzw0);

        double reproj_rms_error_l_0 = CalcRmsReprojError(xy_l_0, P_l, xyzw0);
        double reproj_rms_error_r_0 = CalcRmsReprojError(xy_r_0, P_r, xyzw0);

        Mat_<double> xyzw1;
        dlt.triangulate(ProjectiveCamera(P_l), ProjectiveCamera(P_r), xy_l_1, xy_r_1, xyzw1);

        double reproj_rms_error_l_1 = CalcRmsReprojError(xy_l_1, P_l, xyzw1);
        double reproj_rms_error_r_1 = CalcRmsReprojError(xy_r_1, P_r, xyzw1);

        cout << "\n(F0) DLT reprojection RMS errors (l0 r0 l1 r1) = ("
             << reproj_rms_error_l_0 << " " << reproj_rms_error_r_0 << " "
             << reproj_rms_error_l_1 << " " << reproj_rms_error_r_1 << ")\n";

        // Check if we can find structure using F1 instead of F0

        Mat_<double> P_r_ = Extract2ndCameraMatFromF(F1);

        Mat_<double> xyzw0_;
        dlt.triangulate(ProjectiveCamera(P_l), ProjectiveCamera(P_r_), xy_l_0, xy_r_0, xyzw0_);

        double reproj_rms_error_l_0_ = CalcRmsReprojError(xy_l_0, P_l, xyzw0_);
        double reproj_rms_error_r_0_ = CalcRmsReprojError(xy_r_0, P_r_, xyzw0_);

        Mat_<double> xyzw1_;
        dlt.triangulate(ProjectiveCamera(P_l), ProjectiveCamera(P_r_), xy_l_1, xy_r_1, xyzw1_);

        double reproj_rms_error_l_1_ = CalcRmsReprojError(xy_l_1, P_l, xyzw1_);
        double reproj_rms_error_r_1_ = CalcRmsReprojError(xy_r_1, P_r_, xyzw1_);

        cout << "(F1) DLT reprojection RMS errors (l0 r0 l1 r1) = ("
             << reproj_rms_error_l_0_ << " " << reproj_rms_error_r_0_ << " "
             << reproj_rms_error_l_1_ << " " << reproj_rms_error_r_1_ << ")\n";

        // Match two stereo pairs

        cout << "\nMatching two stereo pairs using left images...";

        vector<DMatch> matches_ll;
        MatchSyntheticShots(*(left_features_collection.find(0)->second),
                            *(left_features_collection.find(1)->second),
                            matches_ll);

        cout << " #matches = " << matches_ll.size() << endl;

//        // Find homography mapping 1st cloud to 2nd one

//        Mat_<double> H12 = FindHomographyLinear(xyzw0, xyzw1);

//        cout << "\nH12 = \n" << H12 << endl;
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
        else if (string(argv[i]) == "--create-images")
            create_images = (bool)atoi(argv[++i]);
        else
            throw runtime_error(string("Can't parse command line arg: ") + argv[i]);
    }
}
