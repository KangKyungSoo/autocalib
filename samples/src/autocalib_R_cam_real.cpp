#pragma warning(disable: 4800)
#include <iostream>
#include <algorithm>
#include <vector>
#include <set>
#include <stdexcept>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <core/include/core.h>

using namespace std;
using namespace cv;
using namespace autocalib;

void ParseArgs(int argc, char **argv);

vector<string> img_names;
vector<Mat> imgs;
int num_frames = 0; // Use all source frames
Ptr<FeaturesFinderCreator> features_finder_creator = new SurfFeaturesFinderCreator();
BestOf2NearestMatcherCreator features_matcher_creator;
FeaturesCollection features_collection;
int min_num_matches = 6;
double H_est_thresh = 3.;
double conf_thresh = 0;
Mat_<double> K_init;
bool lin_est_skew = false;
bool refine_skew = false;
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

        if (img_names.size() < 1)
            throw runtime_error("Need at least one frame");

        for (size_t i = 0; i < img_names.size(); ++i) {
            Mat img = imread(img_names[i]);
            if (img.empty())
                throw runtime_error("Can't open image: " + img_names[i]);
            imgs.push_back(img);
        }

        // Find features

        cout << "\nFinding features...\n";
        Ptr<detail::FeaturesFinder> features_finder = features_finder_creator->Create();

        for (int i = 0; i < num_frames; ++i) {
            int64 t = getTickCount();
            cout << "Finding features in '" << img_names[i] << "'... ";

            Ptr<detail::ImageFeatures> features = new detail::ImageFeatures();
            (*features_finder)(imgs[i], *features);
            features_collection[i] = features;

            cout << "#features = " << features_collection.find(i)->second->keypoints.size()
                 << ", time = " << (getTickCount() - t) / getTickFrequency() << " sec\n";
        }

        // Match all pairs

        cout << "\nMatching pairs... ";
        MatchesCollection matches_collection;
        Ptr<detail::FeaturesMatcher> matcher = features_matcher_creator.Create();

        FeaturesCollection::iterator from_iter = features_collection.begin();
        FeaturesCollection::iterator from_next_iter = from_iter; ++from_next_iter;
        FeaturesCollection::iterator to_iter;

        for (; from_next_iter != features_collection.end(); from_iter = from_next_iter++) {
            for (to_iter = from_next_iter; to_iter != features_collection.end(); ++to_iter) {
                cout << "(" << from_iter->first << "->" << to_iter->first << ") ";
                detail::MatchesInfo mi;
                (*matcher)(*(from_iter->second), *(to_iter->second), mi);
                matches_collection[make_pair(from_iter->first, to_iter->first)]
                        = new vector<DMatch>(mi.matches);
                cout.flush();
            }
        }
        cout << endl;

        // Estimate homographies

        HomographiesP2 Hs;
        RelativeConfidences rel_confs;
        Mat keypoints1, keypoints2;
        double total_confidence = 0;
        double total_init_R_error = 0;

        cout << "\nEstimating Hs...\n";
        for (int from = 0; from < num_frames - 1; ++from) {
            for (int to = from + 1; to < num_frames; ++to) {
                const vector<DMatch> &matches = *(matches_collection.find(make_pair(from, to))->second);

                cout << "Estimating H between '" << img_names[from] << "' and '" << img_names[to]
                     << "'... #matches = " << matches.size();

                if (static_cast<int>(matches.size()) < min_num_matches) {
                    cout << ", not enough matches\n";
                    continue;
                }

                ExtractMatchedKeypoints(*(features_collection.find(from)->second),
                                        *(features_collection.find(to)->second),
                                        matches, keypoints1, keypoints2);
                vector<uchar> inliers_mask;
                Mat_<double> H = findHomography(keypoints1.reshape(2), keypoints2.reshape(2), 
                                                inliers_mask, RANSAC, H_est_thresh);

                if (H.empty()) {
                    cout << ", can't estimate H\n";
                    continue;
                }

                Ptr<vector<DMatch> > inliers = new vector<DMatch>();
                for (size_t i = 0; i < matches.size(); ++i)
                    if (inliers_mask[i])
                        inliers->push_back(matches[i]);
                cout << ", #inliers = " << inliers->size();

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
                cout << ", RMS err = " << rms_err;

                // See "Automatic Panoramic Image Stitching using Invariant Features"
                // by Matthew Brown and David G. Lowe, IJCV 2007 for the explanation
                double confidence = inliers->size() / (8 + 0.3 * matches.size()) - 1;

                cout << ", conf = " << confidence;

                matches_collection[make_pair(from, to)] = inliers;

                if (confidence > conf_thresh) {
                    rel_confs[make_pair(from, to)] = confidence;
                    Hs[make_pair(from, to)] = H;
                    total_confidence += confidence;

                    if (!K_init.empty()) {
                        Mat R = K_init.inv() * H * K_init;
                        R /= pow(abs(determinant(R)), 1. / 3.);
                        double init_R_error = norm(R * R.t(), Mat::eye(3, 3, CV_64F));
                        total_init_R_error += init_R_error;
                        cout << ", |R*R.t()-I| = " << init_R_error << endl;
                    }
                }

                cout << endl;
            }     
        }

        cout << "Avg. confidence = " << total_confidence / Hs.size() << endl;
        if (!K_init.empty())
            cout << "Avg. init R error = " << total_init_R_error / Hs.size() << endl;

        // Find efficient correspondences graph

        detail::Graph eff_corresps;
        int ref_frame_idx = ExtractEfficientCorrespondences(num_frames, rel_confs, eff_corresps);

        // Linear calibration

        double residual_error;
        if (K_init.empty()) {
            cout << "\nLinear calibrating...\n";
            if (lin_est_skew)
                K_init = CalibRotationalCameraLinear(Hs, &residual_error);
            else
                K_init = CalibRotationalCameraLinearNoSkew(Hs, &residual_error);
            cout << "K_init =\n" << K_init << endl;
        }

        RelativeRotationMats rel_Rs;
        for (HomographiesP2::iterator iter = Hs.begin(); iter != Hs.end(); ++iter) {
            Mat R = K_init.inv() * iter->second * K_init;
            SVD svd(R, SVD::FULL_UV);
            rel_Rs[iter->first] = svd.u * svd.vt;
        }

        // Non-linear refinement

        cout << "\nRefining camera...\n";

        AbsoluteRotationMats Rs;
        CalcAbsoluteRotations(rel_Rs, eff_corresps, ref_frame_idx, Rs);

        cout << "The following pairs will be used: \n";
        MatchesCollection eff_matches_collection;
        for (MatchesCollection::iterator iter = matches_collection.begin();
             iter != matches_collection.end(); ++iter)
        {
            if (Rs.find(iter->first.first) != Rs.end() && Rs.find(iter->first.second) != Rs.end() &&
                rel_confs.find(iter->first) != rel_confs.end())
            {
                cout << "'" << img_names[iter->first.first] << "'->'" << img_names[iter->first.second] << "'\n";
                eff_matches_collection[iter->first] = iter->second;
            }
        }

        Mat_<double> K_refined = K_init.clone();
        double final_reproj_error;
        if (refine_skew)
            final_reproj_error = RefineRigidCamera(K_refined, Rs, features_collection, eff_matches_collection);
        else {
            K_refined(0, 1) = 0;
            final_reproj_error = RefineRigidCamera(K_refined, Rs, features_collection, eff_matches_collection,
                                                   ~REFINE_FLAG_SKEW);
        }
        cout << "K_refined =\n" << K_refined << endl;

        cout << "\nSUMMARY\n";
        cout << "K_init =\n" << K_init << endl;
        cout << "K_refined =\n" << K_refined << endl;

        if (!log_file.empty()) {
            ofstream f(log_file.c_str(), ios_base::app);
            f << K_init(0, 0) << ";" << K_init(1, 1) << ";" << K_init(0, 2) << ";" << K_init(1, 2) << ";" << K_init(0, 1) << ";"
              << K_refined(0, 0) << ";" << K_refined(1, 1) << ";" << K_refined(0, 2) << ";" << K_refined(1, 2) << ";" << K_refined(0, 1) << ";"
              << residual_error << ";" << final_reproj_error << ";";
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
        if (string(argv[i]) == "--num-frames")
            num_frames = atoi(argv[++i]);
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
                features_matcher_creator.matcher = new BruteForceMatcher<L1<float> >();
            else if (string(argv[i + 1]) == "bfm_l2")
                features_matcher_creator.matcher = new BruteForceMatcher<L2<float> >();
            else if (string(argv[i + 1]) == "flann")
                features_matcher_creator.matcher = new FlannBasedMatcher();
            else if (string(argv[i + 1]) == "bfm_hamming")
                features_matcher_creator.matcher = new BruteForceMatcher<Hamming>();
            else if (string(argv[i + 1]) == "bfm_hamming_lut")
                features_matcher_creator.matcher = new BruteForceMatcher<HammingLUT>();
            else
                throw runtime_error(string("Unknown matcher type: ") + argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--match-conf")
            features_matcher_creator.match_conf = static_cast<float>(atof(argv[++i]));
        else if (string(argv[i]) == "--min-num-matches")
            min_num_matches = atoi(argv[++i]);
        else if (string(argv[i]) == "--H-est-thresh")
            H_est_thresh = static_cast<float>(atof(argv[++i]));
        else if (string(argv[i]) == "--conf-thresh")
            conf_thresh = static_cast<float>(atof(argv[++i]));
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
        else if (string(argv[i]) == "--log-file")
            log_file = argv[++i];
        else {
            img_names.push_back(argv[i]);
        }
    }
}
