#include <iostream>
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
Ptr<FeaturesFinderCreator> features_finder_creator = new SurfFeaturesFinderCreator();
string features_file;
bool save_features = false;
bool load_features = false;
BestOf2NearestMatcherCreator matcher_creator;
FeaturesCollection features_collection;
int min_num_matches = 6;
double H_est_thresh = 3.;
Mat_<double> K_init;
bool lin_est_skew = false;
bool refine_skew = false;

int main(int argc, char **argv) {
    try {
        ParseArgs(argc, argv);

        int num_frames = static_cast<int>(imgs.size());
        if (num_frames < 1)
            throw runtime_error("Need at least one camera");

        // Find features

        if (!load_features) {
            cout << "Finding features...\n";
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
        }
        else {
            FileStorage f(features_file, FileStorage::READ);
            int num_frames_cached;
            f["num_frames"] >> num_frames_cached;
            CV_Assert(num_frames == num_frames_cached);

            for (int i = 0; i < num_frames_cached; ++i) {
                Ptr<detail::ImageFeatures> features = new detail::ImageFeatures();

                stringstream name;
                name << "keypoints" << i;
                Mat keypoints;
                f[name.str()] >> keypoints;
                features->keypoints.resize(keypoints.rows);
                for (size_t j = 0; j < keypoints.rows; ++j)
                    features->keypoints[j].pt = keypoints.at<Point2f>(j, 0);

                name.str("");
                name << "descriptors" << i;
                f[name.str()] >> features->descriptors;

                features_collection[i] = features;
            }
        }

        if (save_features) {
            FileStorage f(features_file, FileStorage::WRITE);
            f << "num_frames" <<  num_frames;

            for (FeaturesCollection::iterator iter = features_collection.begin();
                 iter != features_collection.end(); ++iter)
            {
                stringstream name;
                name << "keypoints" << iter->first;
                Mat keypoints(iter->second->keypoints.size(), 2, CV_32F);
                for (size_t i = 0; i < iter->second->keypoints.size(); ++i)
                    keypoints.at<Point2f>(i, 0) = iter->second->keypoints[i].pt;
                f << name.str() << keypoints;

                name.str("");
                name << "descriptors" << iter->first;
                f << name.str() << iter->second->descriptors;
            }
        }

        // Match all pairs

        cout << "Matching pairs... ";
        MatchesCollection matches_collection;
        Ptr<detail::FeaturesMatcher> matcher = matcher_creator.Create();

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

        cout << "Estimating Hs...\n";
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
                Mat_<double> H = findHomography(keypoints1, keypoints2, inliers_mask, RANSAC, H_est_thresh);

                if (H.empty()) {
                    cout << ", can't estimate H\n";
                    continue;
                }

                int num_inliers = 0;
                for (size_t i = 0; i < matches.size(); ++i)
                    if (inliers_mask[i])
                        num_inliers++;
                cout << ", #inliers = " << num_inliers;

                double rms_err = 0;
                for (int i = 0; i < keypoints1.cols; ++i) {
                    const Point2f &kp1 = keypoints1.at<Point2f>(0, i);
                    const Point2f &kp2 = keypoints2.at<Point2f>(0, i);
                    double x = H(0, 0) * kp1.x + H(0, 1) * kp1.y + H(0, 2);
                    double y = H(1, 0) * kp1.x + H(1, 1) * kp1.y + H(1, 2);
                    double z = H(2, 0) * kp1.x + H(2, 1) * kp1.y + H(2, 2);
                    x /= z; y /= z;
                    rms_err += (kp2.x - x) * (kp2.x - x) + (kp2.y - y) * (kp2.y - y);
                }
                rms_err = sqrt(rms_err / keypoints1.cols);
                cout << ", RMS err = " << rms_err;

                // See "Automatic Panoramic Image Stitching using Invariant Features"
                // by Matthew Brown and David G. Lowe, IJCV 2007 for the explanation
                double confidence = num_inliers / (8 + 0.3 * matches.size()) - 1;

                cout << ", conf = " << confidence;
                cout << endl;

                if (confidence > 0) {
                    rel_confs[make_pair(from, to)] = confidence;
                    Hs[make_pair(from, to)] = H;
                }
            }
        }

        // Find efficient correspondences graph

        detail::Graph eff_corresps;
        int ref_frame_idx = ExtractEfficientCorrespondences(num_frames, rel_confs, eff_corresps);

        // Linear calibration

        if (K_init.empty()) {
            cout << "Linear calibrating...\n";
            if (lin_est_skew)
                K_init = CalibRotationalCameraLinear(Hs);
            else
                K_init = CalibRotationalCameraLinearNoSkew(Hs);
            cout << "K_init =\n" << K_init << endl;
        }

        RelativeRotationMats rel_Rs;
        for (HomographiesP2::iterator iter = Hs.begin(); iter != Hs.end(); ++iter)
            rel_Rs[iter->first] = K_init.inv() * iter->second * K_init;

        // Non-linear refinement

        cout << "Refining camera...\n";

        AbsoluteRotationMats Rs;
        GetAbsoluteRotations(rel_Rs, eff_corresps, ref_frame_idx, Rs);

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
        if (refine_skew)
            RefineRigidCamera(K_refined, Rs, features_collection, eff_matches_collection);
        else {
            K_refined(0, 1) = 0;
            RefineRigidCamera(K_refined, Rs, features_collection, eff_matches_collection,
                              REFINE_FLAG_ALL & ~REFINE_FLAG_SKEW);
        }
        cout << "K_refined =\n" << K_refined << endl;

        cout << "SUMMARY\n";
        cout << "K_init =\n" << K_init << endl;
        cout << "K_refined =\n" << K_refined << endl;
    }
    catch (const exception &e) {
        cout << "Error: " << e.what() << endl;
    }
    return 0;
}


void ParseArgs(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--features") {
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
        else if (string(argv[i]) == "--save-features") {
            features_file = argv[++i];
            save_features = true;
        }
        else if (string(argv[i]) == "--load-features") {
            features_file = argv[++i];
            load_features = true;
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
        else if (string(argv[i]) == "--H-est-thresh")
            H_est_thresh = static_cast<float>(atof(argv[++i]));
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
        else {
            Mat img = imread(argv[i]);
            if (img.empty())
                throw runtime_error(string("Can't open image: ") + argv[i]);
            img_names.push_back(argv[i]);
            imgs.push_back(img);
        }
    }
}
