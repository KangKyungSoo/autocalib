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

vector<pair<string, string> > img_names;
vector<Mat> left_imgs, right_imgs;
int num_frames = 0; // Use all source frames
Ptr<FeaturesFinderCreator> features_finder_creator = new SurfFeaturesFinderCreator();
BestOf2NearestMatcherCreator matcher_creator;
int min_num_matches = 6;
FeaturesCollection features_collection;
MatchesCollection matches_collection;
Mat_<double> K_init;
double F_est_thresh = 0.3;
double F_est_conf = 0.99;
int H_est_num_iters = 100;
int H_est_subset_size = 10;
double H_est_thresh = 3.;
double conf_thresh = 0;
string log_file;

int main(int argc, char **argv) {
    try {        
        ParseArgs(argc, argv);

        srand(0);

        if (num_frames > 0 && num_frames <= static_cast<int>(img_names.size())) {
            random_shuffle(img_names.begin(), img_names.end());
            img_names.resize(num_frames);
        }
        else
            num_frames = static_cast<int>(img_names.size());

        if (num_frames < 2)
            throw runtime_error("Need at least two frames");

        for (size_t i = 0; i < img_names.size(); ++i) {
            Mat left_img = imread(img_names[i].first);
            if (left_img.empty())
                throw runtime_error("Can't open image: " + img_names[i].first);
            left_imgs.push_back(left_img);

            Mat right_img = imread(img_names[i].second);
            if (right_img.empty())
                throw runtime_error("Can't open image: " + img_names[i].second);
            right_imgs.push_back(right_img);
        }

        // Find features

        cout << "\nFinding features...\n";
        Ptr<detail::FeaturesFinder> features_finder = features_finder_creator->Create();

        for (int i = 0; i < num_frames; ++i) {
            int64 t = getTickCount();
            cout << "Finding features in " << img_names[i].first << "... ";

            Ptr<detail::ImageFeatures> left_features = new detail::ImageFeatures();
            (*features_finder)(left_imgs[i], *left_features);
            features_collection[2 * i] = left_features;

            cout << "#features = " << features_collection.find(2 * i)->second->keypoints.size()
                 << ", time = " << (getTickCount() - t) / getTickFrequency() << " sec\n";

            t = getTickCount();
            cout << "Finding features in " << img_names[i].second << "... ";

            Ptr<detail::ImageFeatures> right_features = new detail::ImageFeatures();
            (*features_finder)(right_imgs[i], *right_features);
            features_collection[2 * i + 1] = right_features;

            cout << "#features = " << features_collection.find(2 * i + 1)->second->keypoints.size()
                 << ", time = " << (getTickCount() - t) / getTickFrequency() << " sec\n";
        }

        // Match everything

        cout << "\nMatch everything...";
        MatchesCollection matches_collection;
        Ptr<detail::FeaturesMatcher> matcher = matcher_creator.Create();

        for (int i = 0; i < num_frames; ++i) {
            cout << "(" << 2 * i << ", " << 2 * i + 1 << ") ";
            detail::MatchesInfo lr_mi;
            (*matcher)(*(features_collection.find(2 * i)->second), *(features_collection.find(2 * i + 1)->second), lr_mi);
            matches_collection[make_pair(2 * i, 2 * i + 1)] = new vector<DMatch>(lr_mi.matches);
            cout.flush();

            for (int j = i + 1; j < num_frames; ++j) {
                cout << "(" << 2 * i << ", " << 2 * j << ") ";
                detail::MatchesInfo ll_mi;
                (*matcher)(*(features_collection.find(2 * i)->second), *(features_collection.find(2 * j)->second), ll_mi);
                matches_collection[make_pair(2 * i, 2 * j)] = new vector<DMatch>(ll_mi.matches);
                cout.flush();
            }
        }

        // Find fundamental matrix and extract camera mat

        cout << "\nFinding F...\n";

        Mat_<double> F = FindFundamentalMatFromPairs(features_collection, matches_collection, F_est_thresh, F_est_conf);
        Mat_<double> P_l = Mat::eye(3, 4, CV_64F);
        Mat_<double> P_r = ExtractCameraMatFromFundamentalMat(F);

        // Remove outliers

        cout << "\nRemoving outliers...\n";

        RelativeConfidences rel_confs;
        MatchesCollection conf_matches_collection;

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
                F_ = findFundamentalMat(xy1.reshape(2), xy2.reshape(2), FM_RANSAC, F_est_thresh);
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

            iter->second = inliers;

            if (conf > conf_thresh) {
                conf_matches_collection[iter->first] = inliers;
                rel_confs[iter->first] = conf;
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
                                                    H_est_num_iters, H_est_subset_size, H_est_thresh,
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

        Mat_<double> K_est, R_est, T_est;

        // Linear autocalibration

        if (K_init.empty()) {
            cout << "\nLinear calibrating...\n";
            K_init = CalibRotationalCameraLinearNoSkew(Hs_inf);
            cout << "K_linear = \n" << K_init << endl;
        }

        cout << "\nK_init = \n" << K_init << endl;

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

        Mat avg_R;
        Rodrigues(total_rvec / total_estimations, avg_R);

        Mat avg_T = total_T / total_estimations;

        RigidCamera P_r_m(K_init, avg_R.clone(), avg_T.clone());

        double final_rms_error = 0;

        detail::Graph eff_corresp(num_frames);
        for (size_t i = 0; i < num_frames - 1; ++i) {
            for (size_t j = i + 1; j < num_frames; ++j) {
                eff_corresp.addEdge(i, j, 0);
                eff_corresp.addEdge(j, i, 0);
            }
        }

        AbsoluteMotions abs_motions;
        CalcAbsoluteMotions(rel_motions, eff_corresp, 0, abs_motions);

        final_rms_error = RefineStereoCamera(P_r_m, abs_motions, features_collection, matches_collection, ~REFINE_FLAG_SKEW);

        cout << "\nK_refined = \n" << P_r_m.K() << endl;

        K_est = P_r_m.K();
        R_est = P_r_m.R().t();
        T_est = -P_r_m.R().t() * P_r_m.T();

        Mat_<double> rvec_est;
        Rodrigues(R_est, rvec_est);
        cout << "rvec est = " << rvec_est << endl;

        cout << "T est = " << T_est << endl;
        cout << "K_est = \n" << K_est << endl;

        if (!log_file.empty()) {
            ofstream f(log_file.c_str(), ios_base::app);
            f << K_init(0, 0) << ";" << K_init(1, 1) << ";" << K_init(0, 2) << ";" << K_init(1, 2) << ";" << K_init(0, 1) << ";"
              << K_est(0, 0) << ";" << K_est(1, 1) << ";" << K_est(0, 2) << ";" << K_est(1, 2) << ";" << K_est(0, 1) << ";"
              << rvec_est(0, 0) << ";" << rvec_est(0, 1) << ";" << rvec_est(0, 2) << ";"
              << T_est(0, 0) << ";" << T_est(1, 0) << ";" << T_est(2, 0) << ";"
              << final_rms_error << ";";
            f << endl;
        }
    }
    catch (const exception &e) {
        cout << "Error: " << e.what() << endl;
    }
    return 0;
}


void ParseArgs(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--num-frames")
            num_frames = atoi(argv[i]);
        else if (string(argv[i]) == "--features") {
            if (string(argv[i + 1]) == "surf")
                features_finder_creator = new SurfFeaturesFinderCreator();
            else if (string(argv[i + 1]) == "orb")
                features_finder_creator = new OrbFeaturesFinderCreator();
            else
                throw runtime_error(string("Unknown features finder type: ") + argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--surf-hess-thresh") {
            FeaturesFinderCreator *ffc = static_cast<FeaturesFinderCreator*>(features_finder_creator);
            SurfFeaturesFinderCreator *sffc = dynamic_cast<SurfFeaturesFinderCreator*>(ffc);
            if (!sffc)
                throw runtime_error(string("Inconsistent features finder option: ") + argv[i + 1]);
            sffc->hess_thresh = atoi(argv[++i]);
        }
        else if (string(argv[i]) == "--surf-num-octaves") {
            FeaturesFinderCreator *ffc = static_cast<FeaturesFinderCreator*>(features_finder_creator);
            SurfFeaturesFinderCreator *sffc = dynamic_cast<SurfFeaturesFinderCreator*>(ffc);
            if (!sffc)
                throw runtime_error(string("Inconsistent features finder option: ") + argv[i + 1]);
            sffc->num_octaves = atoi(argv[++i]);
        }
        else if (string(argv[i]) == "--surf-num-layers") {
            FeaturesFinderCreator *ffc = static_cast<FeaturesFinderCreator*>(features_finder_creator);
            SurfFeaturesFinderCreator *sffc = dynamic_cast<SurfFeaturesFinderCreator*>(ffc);
            if (!sffc)
                throw runtime_error(string("Inconsistent features finder option: ") + argv[i + 1]);
            sffc->num_layers = atoi(argv[++i]);
        }
        else if (string(argv[i]) == "--orb-num-features") {
            FeaturesFinderCreator *ffc = static_cast<FeaturesFinderCreator*>(features_finder_creator);
            OrbFeaturesFinderCreator *offc = dynamic_cast<OrbFeaturesFinderCreator*>(ffc);
            if (!offc)
                throw runtime_error(string("Inconsistent features finder option: ") + argv[i + 1]);
            offc->num_features = atoi(argv[++i]);
        }
        else if (string(argv[i]) == "--matcher") {
            if (string(argv[i + 1]) == "bfm_l1")
                matcher_creator.matcher = new BruteForceMatcher<L1<float> >();
            else if (string(argv[i + 1]) == "bfm_l2")
                matcher_creator.matcher = new BruteForceMatcher<L2<float> >();
            else if (string(argv[i + 1]) == "flann")
                matcher_creator.matcher = new FlannBasedMatcher();
            else if (string(argv[i + 1]) == "bfm_hamming")
                matcher_creator.matcher = new BruteForceMatcher<Hamming>();
            else if (string(argv[i + 1]) == "bfm_hamming_lut")
                matcher_creator.matcher = new BruteForceMatcher<HammingLUT>();
            else
                throw runtime_error(string("Unknown matcher type: ") + argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--match-conf")
            matcher_creator.match_conf = static_cast<float>(atof(argv[++i]));
        else if (string(argv[i]) == "--min-num-matches")
            min_num_matches = atoi(argv[++i]);
        else if (string(argv[i]) == "--K-init") {
            K_init = Mat::eye(3, 3, CV_64F);
            K_init(0, 0) = atof(argv[i + 1]);
            K_init(0, 1) = atof(argv[i + 2]);
            K_init(0, 2) = atof(argv[i + 3]);
            K_init(1, 1) = atof(argv[i + 4]);
            K_init(1, 2) = atof(argv[i + 5]);
            i += 5;
        }
        else if (string(argv[i]) == "--F-est-thresh")
            F_est_thresh = atof(argv[++i]);
        else if (string(argv[i]) == "--F-est-conf")
            F_est_conf = atof(argv[++i]);
        else if (string(argv[i]) == "--H-est-num-iters")
            H_est_num_iters = atof(argv[++i]);
        else if (string(argv[i]) == "--H-est-subset-size")
            H_est_subset_size = atof(argv[++i]);
        else if (string(argv[i]) == "--H-est-thresh")
            H_est_thresh = atof(argv[++i]);
        else if (string(argv[i]) == "--log-file")
            log_file = argv[++i];
        else {
            if (i < argc - 1)
                img_names.push_back(make_pair(argv[i], argv[i + 1]));
            else
                throw runtime_error("Can't find right camera image");
        }
    }
}

