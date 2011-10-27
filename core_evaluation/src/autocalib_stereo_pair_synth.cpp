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
Rect viewport = Rect(0, 0, 1920, 1080);
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

        vector<RigidCamera> left_cameras(num_frames);
        vector<RigidCamera> right_cameras(num_frames);
        FeaturesCollection right_features_collection;
        FeaturesCollection left_features_collection;

        // Generate cameras and shots

        Mat_<double> offset(3, 1);
        offset(0, 0) = 0.1; offset(1, 0) = offset(2, 0) = 0;

        rvec = Mat::zeros(3, 1, CV_64F);
        rvec(0, 0) = rvec(2, 0) = CV_PI / 40; rvec(1, 0) = CV_PI / 20;
        Rodrigues(rvec, R);

        Mat_<double> T(3, 1);
        T(0, 0) = -2; T(1, 0) = 0; T(2, 0) = -10;

        left_cameras[0] = RigidCamera::LocalToWorld(K_gold, R, -R * offset + T);
        right_cameras[0] = RigidCamera::LocalToWorld(K_gold, R, R * offset + T);

        T(0, 0) = 2; T(1, 0) = 0; T(2, 0) = -10;

        left_cameras[1] = RigidCamera::LocalToWorld(K_gold, R.t(), -R.t() * offset + T);
        right_cameras[1] = RigidCamera::LocalToWorld(K_gold, R.t(), R.t() * offset + T);

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
                Mat img;

                CreateImage(*(left_features_collection.find(i)->second), img);
                stringstream name; name << "left_camera" << i << ".jpg";
                imwrite(name.str(), img);

                CreateImage(*(right_features_collection.find(i)->second), img);
                name.str(""); name << "right_camera" << i << ".jpg";
                imwrite(name.str(), img);
            }
        }

        // Find fundmental matrix

        cout << "Finding F between #0 pair images... ";

        vector<DMatch> matches_lr_0;
        MatchSyntheticShots(*(left_features_collection.find(0)->second),
                            *(right_features_collection.find(0)->second),
                            matches_lr_0);

        cout << ", #matches = " << matches_lr_0.size();

        Mat kps_l_0, kps_r_0;
        ExtractMatchedKeypoints(*(left_features_collection.find(0)->second),
                                *(right_features_collection.find(0)->second),
                                matches_lr_0, kps_l_0, kps_r_0);

        vector<uchar> inlier_mask;
        Mat_<double> F = findFundamentalMat(kps_l_0.reshape(2), kps_r_0.reshape(2), inlier_mask);

        int num_inliers = 0;
        for (size_t i = 0; i < inlier_mask.size(); ++i)
            if (inlier_mask[i])
                num_inliers++;

        cout << ", #inliers = " << num_inliers << endl;
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
