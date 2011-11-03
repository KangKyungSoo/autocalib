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
Mat_<double> K_init;
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

        vector<RigidCamera> left_cameras(num_frames);
        vector<RigidCamera> right_cameras(num_frames);
        FeaturesCollection features_collection;
        MatchesCollection matches_collection;

        // Generate cameras and shots       

        Mat_<double> rel_T(3, 1);
        rel_T(0, 0) = 0.5; rel_T(1, 0) = rel_T(2, 0) = 0;

        Mat_<double> T(3, 1);

        detail::ImageFeatures features;

        for (int i = 0; i < num_frames; ++i) {
            while (true) {
                rng.fill(rvec, RNG::NORMAL, 0, 0.2);
                Rodrigues(rvec, R);

                rng.fill(T, RNG::NORMAL, 0, 2);
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

        // Match synthetic shots

        cout << "\nMatching...\n";

        for (size_t i = 0; i < num_frames; ++i) {
            Ptr<vector<DMatch> > matches = new vector<DMatch>();
            MatchSyntheticShots(*(features_collection.find(2 * i)->second),
                                *(features_collection.find(2 * i + 1)->second),
                                *matches);
            matches_collection[make_pair(2 * i, 2 * i + 1)] = matches;
            cout << "(#matches from " << 2 * i << " to " << 2 * i + 1 << " = " << matches->size() << ") ";
        }

        for (size_t i = 0; i < num_frames - 1; ++i) {
            for (size_t j = i + 1; j < num_frames; ++j) {
                Ptr<vector<DMatch> > matches = new vector<DMatch>();
                MatchSyntheticShots(*(features_collection.find(2 * i)->second),
                                    *(features_collection.find(2 * j)->second),
                                    *matches);
                matches_collection[make_pair(2 * i, 2 * j)] = matches;
                cout << "(#matches from " << 2 * i << " to " << 2 * j << " = " << matches->size() << ") ";
            }
        }

        cout << endl;

        // Find fundamental matrix and extract camera mat

        Mat_<double> F = FindBestFundamentalMatFromPairs(features_collection, matches_collection, 0.1);
        Mat_<double> P_r = ExtractCameraMatFromFundamentalMat(F);

        // Linear autocalibration

        if (K_init.empty()) {
            cout << "\nLinear calibrating...\n";

            HomographiesP2 Hs_inf;

            for (size_t i = 0; i < num_frames - 1; ++i) {
                for (size_t j = i + 1; j < num_frames; ++j) {
                    Ptr<vector<DMatch> > matches_lr0 = matches_collection.find(make_pair(2 * i, 2 * i + 1))->second;
                    Ptr<vector<DMatch> > matches_lr1 = matches_collection.find(make_pair(2 * j, 2 * j + 1))->second;
                    Ptr<vector<DMatch> > matches_ll = matches_collection.find(make_pair(2 * i, 2 * j))->second;

                    Mat_<double> xy_l0, xy_r0, xy_l1, xy_r1;
                    ExtractMatchedKeypoints(*(features_collection.find(2 * i)->second),
                                            *(features_collection.find(2 * i + 1)->second), *matches_lr0, xy_l0, xy_r0);
                    ExtractMatchedKeypoints(*(features_collection.find(2 * j)->second),
                                            *(features_collection.find(2 * j + 1)->second), *matches_lr1, xy_l1, xy_r1);

                    Mat_<double> Hpa, H01, xyzw0, xyzw1;
                    Mat_<double> P_r_ = P_r.clone();
                    AffineRectifyStereoCameraByTwoShots(P_r_, xy_l0, xy_r0, xy_l1, xy_r1, matches_lr0, matches_lr1, matches_ll,
                                                        Hpa, H01, xyzw0, xyzw1);

                    Mat_<double> P_l = Mat::eye(3, 4, CV_64F) * Hpa;
                    Hs_inf[make_pair(2 * i, 2 * j)] = Mat(P_l * H01.inv())(Rect(0, 0, 3, 3));

                    // Stereo pair relative rotation can be very close to the identity matrix. That
                    // can lead to numerical instability in K estimation process, so we avoid using those
                    // rotations in the linear autocalibration algorithm.

                    //Hs_inf[make_pair(2 * i, 2 * i + 1)] = P_r_(Rect(0, 0, 3, 3));
                }
            }

            K_init = CalibRotationalCameraLinearNoSkew(Hs_inf);
            cout << "K_linear = \n" << K_init << endl;
        }

//        // Metric rectification

//        cout << "\nMetric rectification...\n";

//        Mat_<double> Ham = Mat::eye(4, 4, CV_64F);
//        Mat Ham_3x3 = Ham(Rect(0, 0, 3, 3));
//        K_init.copyTo(Ham_3x3);

//        H01 = Ham.inv() * H01 * Ham;
//        H01 /= H01(3, 3);

//        cout << "Metric H01 = \n" << H01 << endl;

//        Mat_<double> R01 = H01(Rect(0, 0, 3, 3));
//        Mat_<double> T01 = H01(Rect(3, 0, 1, 3));

//        P_l = P_l * Ham;
//        P_r = P_r * Ham;

//        xyzw0 = Ham.inv() * xyzw0.reshape(xyzw0.cols / 4).t();
//        xyzw1 = Ham.inv() * xyzw1.reshape(xyzw1.cols / 4).t();
//        xyzw0 = Mat(xyzw0.t()).reshape(0, 1);
//        xyzw1 = Mat(xyzw1.t()).reshape(0, 1);

//        cout << "Reprojection RMS error after metric rectification (l0 r0 l1 r1) = ("
//             << CalcRmsReprojectionError(xy_l0, P_l, xyzw0) << " "
//             << CalcRmsReprojectionError(xy_r0, P_r, xyzw0) << " "
//             << CalcRmsReprojectionError(xy_l1, P_l, xyzw1) << " "
//             << CalcRmsReprojectionError(xy_r1, P_r, xyzw1) << ")\n";

//        Mat_<double> F01 = K_init.inv().t() * CrossProductMat(T01) * R01 * K_init.inv();
//        cout << "Point-to-line distance (l0 vs. l1) RMS = " << CalcRmsEpipolarDistance(xy_l1, xy_l0, F01) << endl;

//        // Refine reconstruction

//        cout << "\nRefining metric reconstruction...\n";

//        AbsoluteMotions motions;
//        motions[0] = Motion(Mat::eye(3, 3, CV_64F), Mat::zeros(3, 1, CV_64F));
//        motions[1] = Motion(R01, T01);

//        RigidCamera P_r0_m_ = RigidCamera::FromProjectiveMat(P_r);
//        RigidCamera P_r0_m(K_init, P_r0_m_.R(), P_r0_m_.T());
//        RefineStereoCamera(P_r0_m, motions, features_collection, matches_collection,
//                           ~REFINE_FLAG_SKEW);

//        Mat_<double> K_refined = P_r0_m.K();
//        cout << "K_refined = \n" << K_refined << endl;
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
        else if (string(argv[i]) == "--num-frames")
            num_frames = atoi(argv[++i]);
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
        else if (string(argv[i]) == "--K-init") {
            K_init = Mat::eye(3, 3, CV_64F);
            K_init(0, 0) = atof(argv[i + 1]);
            K_init(0, 1) = atof(argv[i + 2]);
            K_init(0, 2) = atof(argv[i + 3]);
            K_init(1, 1) = atof(argv[i + 4]);
            K_init(1, 2) = atof(argv[i + 5]);
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

