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
void AddNoise();

Ptr<PointCloudSceneCreator> scene_creator = new SphereSceneCreator();
RNG rng;
FeaturesCollection features_collection;
MatchesCollection matches_collection;
int num_points = 1000;
int num_frames = 2;
Rect viewport = Rect(0, 0, 1920, 1080);
Mat_<double> K_gold;
Mat_<double> K_init;
int seed = 0; // No seed
double F_est_thresh = 3.;
double noise_stddev = -1; // No noise
double conf_thresh = 0;
bool create_images = false;
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

        if (seed > 0) {
            rng.state = seed;
            theRNG() = rng;
            srand(seed);
        }

        Ptr<PointCloudScene> scene;

        // Generate synthetic scene

        scene = scene_creator->Create(num_points, rng);
        Mat scene_rvec = Mat::zeros(3, 1, CV_64F);
        rng.fill(scene_rvec, RNG::UNIFORM, -1, 1);
        Mat scene_R;
        Rodrigues(scene_rvec, scene_R);
        scene->set_R(scene_R);

        vector<RigidCamera> left_cameras(num_frames);
        vector<RigidCamera> right_cameras(num_frames);

        // Generate cameras and shots       

        Mat_<double> T_rel(3, 1);
        T_rel(0, 0) = 1; T_rel(1, 0) = T_rel(2, 0) = 0.1;
        Mat_<double> rvec_rel(1, 3);
        rvec_rel(0, 0) = 0; rvec_rel(0, 1) = 0; rvec_rel(0, 2) = 0; // TODO everything is bad when rvec_rel != 0
        Mat R_rel; Rodrigues(rvec_rel, R_rel);

        detail::ImageFeatures features;

        Mat_<double> rvec(1, 3);
        rvec(0, 0) = 0.1; rvec(0, 1) = 0.2; rvec(0, 2) = 0.1;
        Mat R; Rodrigues(rvec, R);
        Mat_<double> T(3, 1);
        T(0, 0) = 0; T(1, 0) = 0; T(2, 0) = -7;

        Mat R_cur = Mat::eye(3, 3, CV_64F);
        for (int i = 0; i < num_frames; ++i) {
            Mat_<double> T_noise(3, 1);
            rng.fill(T_noise, RNG::NORMAL, -0.2, 0.2);
            Mat T_cur_noised = T + T_noise;

            Mat_<double> rvec_noise(1, 3);
            rng.fill(rvec_noise, RNG::NORMAL, -0.2, 0.2);
            Mat R_noise; Rodrigues(rvec_noise, R_noise);
            Mat R_cur_noised = R_cur * R_noise;

            left_cameras[i] = RigidCamera::LocalToWorld(K_gold, R_cur_noised * R_rel, R_cur_noised * (-T_rel + T_cur_noised));
            right_cameras[i] = RigidCamera::LocalToWorld(K_gold, R_cur_noised * R_rel.t(), R_cur_noised * (T_rel + T_cur_noised));

            R_cur *= R;

            scene->TakeShot(left_cameras[i], viewport, features);
            scene->TakeShot(right_cameras[i], viewport, features);
        }

        for (int i = 0; i < num_frames; ++i) {
            Ptr<detail::ImageFeatures> left_features = new detail::ImageFeatures();
            scene->TakeShot(left_cameras[i], viewport, *left_features);
            features_collection[2 * i] = left_features;

            Ptr<detail::ImageFeatures> right_features = new detail::ImageFeatures();
            scene->TakeShot(right_cameras[i], viewport, *right_features);
            features_collection[2 * i + 1] = right_features;            
        }

        // Add noise before F estimation
        AddNoise();

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
            cout.flush();

//            Mat xy1, xy2;
//            ExtractMatchedKeypoints(*(features_collection.find(2 * i)->second),
//                                    *(features_collection.find(2 * i + 1)->second),
//                                    *matches, xy1, xy2);
//            Mat F_ = findFundamentalMat(xy2.reshape(2), xy1.reshape(2));
//            cout << F_ << endl;

//            Mat P1 = Mat::eye(3, 4, CV_64F), P2;
//            P2 = ExtractCameraMatFromFundamentalMat(F_);

//            cout << P2 << endl;

//            Mat xyzw;
//            DltTriangulation dlt;
//            dlt.triangulate(ProjectiveCamera(P1), ProjectiveCamera(P2), xy1, xy2, xyzw);

//            cout << CalcRmsReprojectionError(xy1, P1, xyzw) << " " << CalcRmsReprojectionError(xy2, P2, xyzw) << endl;
        }
        for (size_t i = 0; i < num_frames - 1; ++i) {
            for (size_t j = i + 1; j < num_frames; ++j) {
                Ptr<vector<DMatch> > matches = new vector<DMatch>();
                MatchSyntheticShots(*(features_collection.find(2 * i)->second),
                                    *(features_collection.find(2 * j)->second),
                                    *matches);
                matches_collection[make_pair(2 * i, 2 * j)] = matches;
                cout << "(#matches from " << 2 * i << " to " << 2 * j << " = " << matches->size() << ") ";
                cout.flush();
            }
        }
        cout << endl;

        // Find fundamental matrix and extract camera mat

        cout << "\nFinding F...\n";

        Mat_<double> F = FindFundamentalMatFromPairs(features_collection, matches_collection, F_est_thresh);
        //Mat_<double> F = K_gold.inv().t() * CrossProductMat(2 * T_rel) * K_gold.inv();
        Mat_<double> P_l = /*RigidCamera::LocalToWorld(K_gold, Mat::eye(3, 3, CV_64F), Mat::zeros(3, 1, CV_64F)).P();*/Mat::eye(3, 4, CV_64F);
        Mat_<double> P_r = /*RigidCamera::LocalToWorld(K_gold, Mat::eye(3, 3, CV_64F), 2 * T_rel).P();*/ExtractCameraMatFromFundamentalMat(F);

        // Remove outliers

        cout << "\nRemoving outliers...\n";

        RelativeConfidences rel_confs;
        MatchesCollection inliers_collection;

        for (MatchesCollection::iterator iter = matches_collection.begin();
             iter != matches_collection.end(); ++iter)
        {
            int from = iter->first.first;
            int to = iter->first.second;

            Ptr<vector<DMatch> > matches = iter->second;
            Mat F_;

            if (IsLeftRightPair(from, to))
                F_ = F;
            else if (BothAreLeft(from, to)) {
                Mat xy1, xy2;
                ExtractMatchedKeypoints(*(features_collection.find(from)->second),
                                        *(features_collection.find(to)->second),
                                        *matches, xy1, xy2);
                F_ = findFundamentalMat(xy2.reshape(2), xy1.reshape(2), FM_RANSAC, F_est_thresh);
            }
            else {
                stringstream msg;
                msg << "from=" << from << ", to=" << to << " - bad matches";
                throw runtime_error(msg.str());
            }

            Mat_<uchar> mask;
            int num_inliers = FindFundamentalMatInliers(*(features_collection.find(from)->second),
                                                        *(features_collection.find(to)->second),
                                                        *matches, F_, F_est_thresh, mask);

            // See "Automatic Panoramic Image Stitching using Invariant Features"
            // by Matthew Brown and David G. Lowe, IJCV 2007 for the explanation
            double conf = num_inliers / (8 + 0.3 * matches->size()) - 1;

            cout << "from=" << from << ", to=" << to << ", #matches=" << matches->size()
                 << ", #inliers=" << num_inliers << ", conf=" << conf << endl;

            Ptr<vector<DMatch> > inliers = new vector<DMatch>();
            inliers->reserve(num_inliers);
            for (size_t i = 0; i < matches->size(); ++i)
                if (mask(0, i))
                    inliers->push_back((*matches)[i]);

            iter->second = matches;

            if (conf > conf_thresh) {
                inliers_collection[iter->first] = inliers;
                if (BothAreLeft(from, to))
                    rel_confs[make_pair(from / 2, to / 2)] = conf;
            }
        }

        // Affine rectification

        map<pair<int, int>, Mat> Ps_l_a;
        map<pair<int, int>, Mat> Ps_r_a;
        HomographiesP2 Hs_inf;
        HomographiesP3 Hs_01_a;

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

                Mat_<double> Hpa, H01_a;
                Mat_<double> xyzw0_a, xyzw1_a;
                Mat_<double> P_l_a_ = P_l.clone();
                Mat_<double> P_r_a_ = P_r.clone();

                AffineRectifyStereoCameraByTwoShots(P_l_a_, P_r_a_, xy_l0, xy_r0, xy_l1, xy_r1, matches_lr0, matches_lr1, matches_ll,
                                                    Hpa, H01_a, xyzw0_a, xyzw1_a);

                Hs_01_a[make_pair(i, j)] = H01_a;

                Ps_l_a[make_pair(i, j)] = P_l_a_;
                Ps_r_a[make_pair(i, j)] = P_r_a_;

                // Stereo pair relative rotation can be very close to the identity matrix. That
                // can lead to numerical instability in K estimation process, so we avoid using those
                // rotations in the linear autocalibration algorithm.

                Hs_inf[make_pair(2 * i, 2 * j)] = Mat(P_l_a_ * H01_a.inv())(Rect(0, 0, 3, 3));
                //Hs_inf[make_pair(2 * i, 2 * i + 1)] = P_r_a_(Rect(0, 0, 3, 3));
            }
        }

        // Linear autocalibration

        double residual_error;
        if (K_init.empty()) {
            cout << "\nLinear calibrating...\n";
            K_init = CalibRotationalCameraLinearNoSkew(Hs_inf, &residual_error);
            cout << "K_linear = \n" << K_init << endl;
        }        

        // Metric rectification

        cout << "\nMetric rectification...\n";

        Mat_<double> Ham = Mat::eye(4, 4, CV_64F);
        Mat Ham_3x3 = Ham(Rect(0, 0, 3, 3));
        K_init.copyTo(Ham_3x3);

        RelativeMotions rel_motions;

        int total_estimations = 0;
        Mat_<double> total_rvec = Mat::zeros(3, 1, CV_64F);
        Mat_<double> total_T = Mat::zeros(3, 1, CV_64F);

        for (HomographiesP3::iterator iter = Hs_01_a.begin(); iter != Hs_01_a.end(); ++iter) {
            Mat H01_a = iter->second;
            Mat H01_m = Ham.inv() * H01_a * Ham;
            H01_m /= H01_m.at<double>(3, 3);
            iter->second = H01_m;

            Mat R01 = H01_m(Rect(0, 0, 3, 3));
            Mat T01 = H01_m(Rect(3, 0, 1, 3));
            rel_motions[iter->first] = Motion(R01, T01);

            RigidCamera rigid_cam = RigidCamera::FromProjectiveMat(Ps_r_a[iter->first] * Ham);

            Mat rvec;
            Rodrigues(rigid_cam.R(), rvec);
            total_rvec += rvec;

            total_T += rigid_cam.T();
            total_estimations++;
        }

        detail::Graph eff_corresp(num_frames);
        for (size_t i = 0; i < num_frames - 1; ++i) {
            for (size_t j = i + 1; j < num_frames; ++j) {
                eff_corresp.addEdge(i, j, 0);
                eff_corresp.addEdge(j, i, 0);
            }
        }

        AbsoluteMotions abs_motions;
        CalcAbsoluteMotions(rel_motions, eff_corresp, 0, abs_motions);

        Mat avg_R;
        Rodrigues(total_rvec / total_estimations, avg_R);
        RigidCamera P_r_m(K_init, avg_R, total_T / total_estimations);

        double final_rms_error = RefineStereoCamera(P_r_m, abs_motions, features_collection, matches_collection,
                                                    ~REFINE_FLAG_SKEW);

        Mat_<double> K_refined = P_r_m.K();
        cout << "K_refined = \n" << K_refined << endl;

        if (!log_file.empty()) {
            ofstream f(log_file.c_str(), ios_base::app);
            f << K_gold(0, 0) << ";" << K_gold(1, 1) << ";" << K_gold(0, 2) << ";" << K_gold(1, 2) << ";" << K_gold(0, 1) << ";"
              << num_frames << ";" << noise_stddev << ";" << F_est_thresh << ";"
              << K_init(0, 0) << ";" << K_init(1, 1) << ";" << K_init(0, 2) << ";" << K_init(1, 2) << ";" << K_init(0, 1) << ";"
              << K_refined(0, 0) << ";" << K_refined(1, 1) << ";" << K_refined(0, 2) << ";" << K_refined(1, 2) << ";" << K_refined(0, 1) << ";"
              << residual_error << ";" << final_rms_error << ";";
            for (int i = 0; i < argc; ++i)
                f << argv[i] << " ";
            f << ";\n";
        }
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
        else if (string(argv[i]) == "--F-est-thresh")
            F_est_thresh = atof(argv[++i]);
        else if (string(argv[i]) == "--noise-stddev")
            noise_stddev = atof(argv[++i]);
        else if (string(argv[i]) == "--create-images")
            create_images = (bool)atoi(argv[++i]);
        else if (string(argv[i]) == "--log-file")
            log_file = argv[++i];
        else
            throw runtime_error(string("Can't parse command line arg: ") + argv[i]);
    }
}


void AddNoise() {
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
}
