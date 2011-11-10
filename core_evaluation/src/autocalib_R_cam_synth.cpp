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

Ptr<IPointCloudSceneCreator> scene_creator = new SphereSceneCreator();
int num_points = 1000;
int num_frames = 5;
Rect viewport = Rect(0, 0, 1920, 1080);
Mat_<double> K_gold;
Mat_<double> K_init;
bool lin_est_skew = false;
bool refine_skew = false;
int seed = 0; // No seed
Mat_<double> camera_center;
double max_angle = 0.1;
bool create_images = false;
double H_est_thresh = 3;
double noise_stddev = -1; // No noise
double noise_transl = -1; // No noise
string log_file;

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

        if (camera_center.empty()) {
            camera_center = Mat::zeros(3, 1, CV_64F);
            camera_center(2, 0) = -10;
        }

        RNG rng;
        if (seed > 0)
            rng.state = seed;

        Mat rvec, R, T;
        Ptr<ISyntheticScene> scene;

        // Generate synthetic scene
        if (!scene_creator.empty()) {
            Ptr<PointCloudScene> scene_ = scene_creator->Create(num_points, rng);
            rvec = Mat::zeros(3, 1, CV_64F);
            rng.fill(rvec, RNG::UNIFORM, -1, 1);
            Rodrigues(rvec, R);
            scene_->set_R(R);
            T = Mat::zeros(3, 1, CV_64F);
            rng.fill(T, RNG::UNIFORM, -2, 2);
            scene_->set_T(T);

            scene = static_cast<ISyntheticScene*>(scene_);
            scene_.addref();
        }
        else {
            CompositeSceneBuilder scene_builder;

            Ptr<PointCloudScene> sphere = new SphereScene(num_points, rng);
            rvec = Mat::zeros(3, 1, CV_64F);
            rng.fill(rvec, RNG::UNIFORM, -1, 1);
            Rodrigues(rvec, R);
            sphere->set_R(R);
            T = Mat::zeros(3, 1, CV_64F);
            rng.fill(T, RNG::UNIFORM, -2, 2);
            sphere->set_T(T);
            scene_builder.Add(sphere);

            Ptr<PointCloudScene> cube = new CubeScene(num_points, rng);
            rvec = Mat::zeros(3, 1, CV_64F);
            rng.fill(rvec, RNG::UNIFORM, -1, 1);
            Rodrigues(rvec, R);
            cube->set_R(R);
            T = Mat::zeros(3, 1, CV_64F);
            rng.fill(T, RNG::UNIFORM, -2, 2);
            cube->set_T(T);
            scene_builder.Add(cube);

            Ptr<CompositeScene> scene_ = scene_builder.Build();
            scene = static_cast<ISyntheticScene*>(scene_);
            scene_.addref();
        }

        vector<RigidCamera> cameras(num_frames);
        FeaturesCollection features_collection;

        // Generate cameras and shots
        for (int i = 0; i < num_frames; ++i) {
            rvec = Mat::zeros(3, 1, CV_64F);
            rng.fill(rvec, RNG::UNIFORM, -1, 1);
            rvec /= norm(rvec) / ((double)rng * max_angle);

            Mat T_noise = Mat::zeros(3, 1, CV_64F);
            if (noise_transl > 0)
                rng.fill(T_noise, RNG::NORMAL, 0, noise_transl);

            Rodrigues(rvec, R);
            cameras[i] = RigidCamera::FromLocalToWorld(K_gold, R, camera_center + T_noise);

            Ptr<detail::ImageFeatures> features = new detail::ImageFeatures();
            scene->TakeShot(cameras[i], viewport, *features);
            features_collection[i] = features;
        }

        if (noise_stddev > 0) {
            cout << "\nAdding noise...\n";
            for (int i = 0; i < num_frames; ++i) {
                Mat_<float> noise(1, 2 * features_collection.find(i)->second->keypoints.size());

                // Final noise RMS is determined by sqrt(noise_x^2 + noise_y^2),
                // so we split by sqrt(2) to get desired noise
                rng.fill(noise, RNG::NORMAL, 0, noise_stddev / sqrt(2.));

                double total_noise = 0;
                Ptr<detail::ImageFeatures> features = features_collection.find(i)->second;
                for (size_t j = 0; j < features->keypoints.size(); ++j) {
                    features->keypoints[j].pt.x += noise(0, 2 * j);
                    features->keypoints[j].pt.y += noise(0, 2 * j + 1);
                    total_noise += noise(0, 2 * j) * noise(0, 2 * j) + noise(0, 2 * j + 1) * noise(0, 2 * j + 1);
                }
                cout << "Shot " << i << " noise RMS error = " << sqrt(total_noise / features->keypoints.size()) << endl;
            }
        }

        if (create_images) {
            for (int i = 0; i < num_frames; ++i) {
                Mat img = CreateImage(*(features_collection.find(i)->second));

                stringstream name;
                name << "camera" << i << ".jpg";
                imwrite(name.str(), img);
            }
        }

        HomographiesP2 Hs;
        vector<Mat> Hs_from_0;
        Mat keypoints1, keypoints2;
        vector<DMatch> matches;
        MatchesCollection matches_collection;

        cout << "\nFinding homographies...\n";
        for (int from = 0; from < num_frames - 1; ++from) {
            for (int to = from + 1; to < num_frames; ++to) {
                MatchSyntheticShots(*(features_collection.find(from)->second),
                                    *(features_collection.find(to)->second),
                                    matches);
                ExtractMatchedKeypoints(*(features_collection.find(from)->second),
                                        *(features_collection.find(to)->second),
                                          matches, keypoints1, keypoints2);

                Mat_<uchar> mask;
                Mat_<double> H = findHomography(keypoints1.reshape(2), keypoints2.reshape(2),
                                                mask, cv::RANSAC, H_est_thresh);

                if (H.empty())
                    cout << "Can't find H from " << from << " to " << to << endl;
                else {
                    Ptr<vector<DMatch> > inlier_matches = new vector<DMatch>();

                    // Put inlier matches into matches collection
                    for (size_t i = 0; i < matches.size(); ++i)
                        if (mask(0, i))
                            inlier_matches->push_back(matches[i]);
                    matches_collection[make_pair(from, to)] = inlier_matches;

                    Hs[make_pair(from, to)] = H;
                    if (from == 0)
                        Hs_from_0.push_back(H.clone());

                    // Compute homography reprojection error
                    double rms_err = 0;
                    for (size_t i = 0; i < matches.size(); ++i) {
                        const Point2d &kp1 = keypoints1.at<Point2d>(0, i);
                        const Point2d &kp2 = keypoints2.at<Point2d>(0, i);
                        double x = H(0, 0) * kp1.x + H(0, 1) * kp1.y + H(0, 2);
                        double y = H(1, 0) * kp1.x + H(1, 1) * kp1.y + H(1, 2);
                        double z = H(2, 0) * kp1.x + H(2, 1) * kp1.y + H(2, 2);
                        x /= z; y /= z;
                        rms_err += (kp2.x - x) * (kp2.x - x) + (kp2.y - y) * (kp2.y - y);
                    }
                    rms_err = sqrt(rms_err / matches.size());
                    cout << "H from " << from << " to " << to
                         << " RMS error = " << sqrt(rms_err / matches.size())
                         << " = sqrt(2) * " << sqrt(rms_err / matches.size()) / sqrt(2.) << endl;

                }
            }
        }

        int64 calib_start_time = getTickCount();

        if (K_init.empty()) {
            cout << "\nLinear calibrating...\n";
            if (lin_est_skew)
                K_init = CalibRotationalCameraLinear(Hs);
            else
                K_init = CalibRotationalCameraLinearNoSkew(Hs);
            cout << "Linear calibration result'll be used as K_init\n";
        }
        cout << "K_init =\n" << K_init << endl;
        
        cout << "\nRefining camera...\n";
        if (Hs_from_0.size() != num_frames - 1) {
            stringstream msg;
            msg << "Refinement requires Hs between first and all other images, "
                << "but only " << Hs_from_0.size() << " were/was found";
            throw runtime_error(msg.str());
        }

        // Refine camera parameters
        map<int, Mat> Rs;
        Rs[0] = Mat::eye(3, 3, CV_64F);
        for (int i = 1; i < num_frames; ++i)
            Rs[i] = K_init.inv() * Hs_from_0[i - 1] * K_init;

        Mat_<double> K_refined = K_init.clone();
        if (refine_skew)
            RefineRigidCamera(K_refined, Rs, features_collection, matches_collection);
        else {
            K_refined(0, 1) = 0;
            RefineRigidCamera(K_refined, Rs, features_collection, matches_collection,
                              ~REFINE_FLAG_SKEW);
        }
        cout << "K_refined =\n" << K_refined << endl;

        int64 calib_time = getTickCount() - calib_start_time;

        cout << "\nSUMMARY\n";
        cout << "K_gold =\n" << K_gold << endl;
        cout << "K_init =\n" << K_init << endl;
        cout << "K_refined =\n" << K_refined << endl;
        cout << "calibration time = " << fixed << setprecision(3)
             << calib_time / getTickFrequency() << " sec\n";

        if (!log_file.empty()) {
            ofstream f(log_file.c_str(), ios_base::app);
            if (!f.is_open())
                throw runtime_error("Can't open log file: " + log_file);
            f << num_points << " " << num_frames << " " << noise_stddev << " ";
            f << K_init(0, 0) << " " << K_init(1, 1) << " " << K_init(0, 2) << " " << K_init(1, 2) << " " << K_init(0, 1) << " ";
            f << K_refined(0, 0) << " " << K_refined(1, 1) << " " << K_refined(0, 2) << " " << K_refined(1, 2) << " " << K_refined(0, 1) << " ";
            f << K_gold(0, 0) << " " << K_gold(1, 1) << " " << K_gold(0, 2) << " " << K_gold(1, 2) << " " << K_gold(0, 1) << " ";
            f << fixed << setprecision(3) << calib_time / getTickFrequency() << " ";
            f << endl;
        }
    }
    catch (const exception &e) {
        cout << "Error: " << e.what() << "\n";
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
            else if (string(argv[i + 1]) == "both")
                scene_creator = 0;
            else
                throw runtime_error(string("Unknown synthetic scene type: ") + argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--num-points")
            num_points = atoi(argv[++i]);
        else if (string(argv[i]) == "--num-frames")
            num_frames = atoi(argv[++i]);
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
        else if (string(argv[i]) == "--lin-est-skew")
            lin_est_skew = atoi(argv[++i]);
        else if (string(argv[i]) == "--refine-skew")
            refine_skew = atoi(argv[++i]);
        else if (string(argv[i]) == "--seed")
            seed = atoi(argv[++i]);
        else if (string(argv[i]) == "--camera-center") {
            camera_center = Mat::zeros(3, 1, CV_64F);
            camera_center(0, 0) = atof(argv[i + 1]);
            camera_center(1, 0) = atof(argv[i + 2]);
            camera_center(2, 0) = atof(argv[i + 3]);
            i += 3;
        }
        else if (string(argv[i]) == "--max-angle")
            max_angle = atof(argv[++i]);
        else if (string(argv[i]) == "--create-images")
            create_images = (bool)atoi(argv[++i]);
        else if (string(argv[i]) == "--H-est-thresh")
            H_est_thresh = atof(argv[++i]);
        else if (string(argv[i]) == "--noise-stddev")
            noise_stddev = atof(argv[++i]);
        else if (string(argv[i]) == "--noise-transl")
            noise_transl = atof(argv[++i]);
        else if (string(argv[i]) == "--log-file")
            log_file = argv[++i];
        else
            throw runtime_error(string("Can't parse command line arg: ") + argv[i]);
    }
}
