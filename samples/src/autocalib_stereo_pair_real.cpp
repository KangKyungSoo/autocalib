#pragma warning(disable: 4800)
#include <numeric>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
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
Size work_size(0, 0);
int blur_ksize = 3;
int num_frames = 0; // Use all source frames
bool manual_registr;
bool save_keypoints, load_keypoints;
bool save_matches, load_matches;
Ptr<FeaturesFinderCreator> features_finder_creator = new SurfFeaturesFinderCreator();
BestOf2NearestMatcherCreator features_matcher_creator;
bool show_matches;
bool opt_flow_matching;
bool opt_assignment_matching;
int min_num_matches = 6;
FeaturesCollection features_collection;
MatchesCollection matches_collection;
Mat_<double> K_init;
double F_est_thresh = 5;
double F_est_conf = 0.99;
int H_est_num_iters = 100;
int H_est_subset_size = 5;
double H_est_thresh = 3.;
double conf_thresh = 1;
string log_file;
string intrinsics_file;
Mat_<double> K1_gold, K2_gold;
Mat_<double> dist1, dist2;
string extrinsics_file;
Mat_<double> R_gold, T_gold;
Mat_<double> F_gold;

int main(int argc, char **argv) {
    try {        
        FeaturesFinderCreator* ffc = static_cast<FeaturesFinderCreator*>(features_finder_creator);
        dynamic_cast<SurfFeaturesFinderCreator*>(ffc)->hess_thresh = 50.;

        features_matcher_creator.match_conf = 0.2f;

        ParseArgs(argc, argv);

        if (!intrinsics_file.empty()) {
            FileStorage fs(intrinsics_file, FileStorage::READ);
            fs["M1"] >> K1_gold;
            fs["M2"] >> K2_gold;
            //fs["D1"] >> dist1;
            //fs["D2"] >> dist2;
        }

        if (!extrinsics_file.empty()) {
            FileStorage fs(extrinsics_file, FileStorage::READ);
            fs["R"] >> R_gold;
            fs["T"] >> T_gold;
        }


        if (!intrinsics_file.empty() && !extrinsics_file.empty()) {
            F_gold = K2_gold.inv().t() * CrossProductMat(T_gold) * R_gold * K1_gold.inv();
            F_gold /= F_gold(2, 2);
            //F_gold = F_gold.t();
        }        

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
//            if (!dist1.empty()) {
//                Mat tmp;
//                undistort(left_img, tmp, K1_gold, dist1, K1_gold);
//                left_img = tmp;
//            }
            left_imgs.push_back(left_img);

            Mat right_img = imread(img_names[i].second);
            if (right_img.empty())
                throw runtime_error("Can't open image: " + img_names[i].second);
//            if (!dist2.empty()) {
//                Mat tmp;
//                undistort(right_img, tmp, K2_gold, dist2, K2_gold);
//                right_img = tmp;
//            }
            right_imgs.push_back(right_img);
        }

        if (blur_ksize > 0) {
            for (int i = 0; i < num_frames; ++i) {
                medianBlur(left_imgs[i], left_imgs[i], blur_ksize);
                medianBlur(right_imgs[i], right_imgs[i], blur_ksize);
            }
        }

        if (!K1_gold.empty() && work_size.width > 0 && work_size.height > 0) {
            K1_gold(0,0) *= work_size.width / (double)left_imgs[0].cols;
            K1_gold(0,1) *= work_size.width / (double)left_imgs[0].cols;
            K1_gold(0,2) *= work_size.width / (double)left_imgs[0].cols;
            K1_gold(1,1) *= work_size.height / (double)left_imgs[0].rows;
            K1_gold(1,2) *= work_size.height / (double)left_imgs[0].rows;
        }

        if (!K2_gold.empty() && work_size.width > 0 && work_size.height > 0) {
            K2_gold(0,0) *= work_size.width / (double)right_imgs[0].cols;
            K2_gold(0,1) *= work_size.width / (double)right_imgs[0].cols;
            K2_gold(0,2) *= work_size.width / (double)right_imgs[0].cols;
            K2_gold(1,1) *= work_size.height / (double)right_imgs[0].rows;
            K2_gold(1,2) *= work_size.height / (double)right_imgs[0].rows;
        }

        if (work_size.width > 0 && work_size.height > 0) {
            for (size_t i = 0; i < left_imgs.size(); ++i)
                resize(left_imgs[i], left_imgs[i], work_size);
            for (size_t i = 0; i < right_imgs.size(); ++i)
                resize(right_imgs[i], right_imgs[i], work_size);
        }

        if (manual_registr) {
            map<int, Ptr<vector<Point2f> > > keypoints;

            if (load_keypoints) {
                for (int i = 0; i < num_frames; ++i) {
                    ifstream fl((img_names[i].first + ".txt").c_str());
                    if (!fl.is_open())
                        throw runtime_error("Can't open " + img_names[i].first + ".txt");
                    Ptr<vector<Point2f> > left_keypoints = new vector<Point2f>();
                    while (!fl.eof()) {
                        float x, y;
                        fl >> x >> y;
                        left_keypoints->push_back(Point2f(x, y));
                    }
                    keypoints[2 * i] = left_keypoints;
                    fl.close();

                    ifstream fr((img_names[i].second + ".txt").c_str());
                    if (!fr.is_open())
                        throw runtime_error("Can't open " + img_names[i].second + ".txt");
                    Ptr<vector<Point2f> > right_keypoints = new vector<Point2f>();
                    while (!fr.eof()) {
                        float x, y;
                        fr >> x >> y;
                        right_keypoints->push_back(Point2f(x, y));
                    }
                    keypoints[2 * i + 1] = right_keypoints;
                    fr.close();
                }
            }
            else {
                for (int i = 0; i < num_frames; ++i) {
                    Ptr<vector<Point2f> > left_keypoints = new vector<Point2f>();
                    the_keypoints_extractor().set_image(left_imgs[i]);
                    the_keypoints_extractor().set_keypoints_output(left_keypoints);
                    the_keypoints_extractor().Run();
                    keypoints[2 * i] = left_keypoints;

                    Ptr<vector<Point2f> > right_keypoints = new vector<Point2f>();
                    the_keypoints_extractor().set_image(right_imgs[i]);
                    the_keypoints_extractor().set_keypoints_output(right_keypoints);
                    the_keypoints_extractor().Run();
                    keypoints[2 * i + 1] = right_keypoints;
                }
            }

            if (save_keypoints) {
                for (int i = 0; i < num_frames; ++i) {
                    Ptr<vector<Point2f> > left_keypoints = keypoints[2 * i];
                    ofstream fl((img_names[i].first + ".txt").c_str());
                    for (size_t j = 0; j < left_keypoints->size(); ++j) {
                        const Point2f &pt = (*left_keypoints)[j];
                        fl << pt.x << " " << pt.y << endl;
                    }
                    fl.close();

                    Ptr<vector<Point2f> > right_keypoints = keypoints[2 * i + 1];
                    ofstream fr((img_names[i].second + ".txt").c_str());
                    for (size_t j = 0; j < right_keypoints->size(); ++j) {
                        const Point2f &pt = (*right_keypoints)[j];
                        fr << pt.x << " " << pt.y << endl;
                    }
                    fr.close();
                }
            }

            for (int i = 0; i < num_frames; ++i) {
                Ptr<detail::ImageFeatures> left_features = new detail::ImageFeatures();
                Ptr<vector<Point2f> > left_keypoints = keypoints.find(2 * i)->second;
                for (size_t j = 0; j < left_keypoints->size(); ++j)
                    left_features->keypoints.push_back(KeyPoint((*left_keypoints)[j], 0));
                features_collection[2 * i] = left_features;

                Ptr<detail::ImageFeatures> right_features = new detail::ImageFeatures();
                Ptr<vector<Point2f> > right_keypoints = keypoints.find(2 * i + 1)->second;
                for (size_t j = 0; j < right_keypoints->size(); ++j)
                    right_features->keypoints.push_back(KeyPoint((*right_keypoints)[j], 0));
                features_collection[2 * i + 1] = right_features;
            }

            for (int i = 0; i < num_frames; ++i) {
                Ptr<vector<Point2f> > keypoints_l = keypoints.find(2 * i)->second;
                Ptr<vector<Point2f> > keypoints_r = keypoints.find(2 * i + 1)->second;
                Ptr<vector<DMatch> > matches_lr = new vector<DMatch>();
                if (load_matches) {
                    string name = img_names[i].first + "_to_" + img_names[i].second + ".txt";
                    ifstream f(name.c_str());
                    if (!f.is_open())
                        throw runtime_error("Can't open " + name);
                    while (!f.eof()) {
                        int from, to;
                        f >> from >> to;
                        matches_lr->push_back(DMatch(from, to, 0));
                    }
                }
                else {
                    the_features_matcher().set_1st_image(left_imgs[i], *keypoints_l);
                    the_features_matcher().set_2nd_image(right_imgs[i], *keypoints_r);
                    the_features_matcher().set_matches_output(matches_lr);
                    the_features_matcher().Run();
                    if (save_matches) {
                        ofstream f((img_names[i].first + "_to_" + img_names[i].second + ".txt").c_str());
                        for (size_t j = 0; j < matches_lr->size(); ++j) {
                            f << (*matches_lr)[j].queryIdx << " " << (*matches_lr)[j].trainIdx << endl;
                        }
                    }
                }
                matches_collection[make_pair(2 * i, 2 * i + 1)] = matches_lr;

                for (int j = i + 1; j < num_frames; ++j) {
                    keypoints_r = keypoints.find(2 * j)->second;
                    the_features_matcher().set_2nd_image(left_imgs[j], *keypoints_r);
                    Ptr<vector<DMatch> > matches_ll = new vector<DMatch>();
                    if (load_matches) {
                        string name = img_names[i].first + "_to_" + img_names[j].first + ".txt";
                        ifstream f(name.c_str());
                        if (!f.is_open())
                            throw runtime_error("Can't open " + name);
                        while (!f.eof()) {
                            int from, to;
                            f >> from >> to;
                            matches_ll->push_back(DMatch(from, to, 0));
                        }
                    }
                    else {
                        the_features_matcher().set_matches_output(matches_ll);
                        the_features_matcher().Run();
                        if (save_matches) {
                            ofstream f((img_names[i].first + "_to_" + img_names[j].first + ".txt").c_str());
                            for (size_t k = 0; k < matches_ll->size(); ++k) {
                                f << (*matches_ll)[k].queryIdx << " " << (*matches_ll)[k].trainIdx << endl;
                            }
                        }
                    }
                    matches_collection[make_pair(2 * i, 2 * j)] = matches_ll;
                }
            }
        }
        else {
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

                if (!opt_flow_matching) {
                    t = getTickCount();
                    cout << "Finding features in " << img_names[i].second << "... ";

                    Ptr<detail::ImageFeatures> right_features = new detail::ImageFeatures();
                    (*features_finder)(right_imgs[i], *right_features);
                    features_collection[2 * i + 1] = right_features;

                    cout << "#features = " << features_collection.find(2 * i + 1)->second->keypoints.size()
                         << ", time = " << (getTickCount() - t) / getTickFrequency() << " sec\n";
                }
            }

            // Match everything

            cout << "\nMatch everything... ";            
            Ptr<detail::FeaturesMatcher> matcher;
            if (opt_assignment_matching) {
                matcher = OptAssignmentMatcherCreator().Create();
            }
            else {
                matcher = features_matcher_creator.Create();
            }

            for (int i = 0; i < num_frames; ++i) {
                if (!opt_flow_matching) {
                    detail::MatchesInfo lr_mi;
                    (*matcher)(*(features_collection.find(2 * i)->second), *(features_collection.find(2 * i + 1)->second), lr_mi);
                    matches_collection[make_pair(2 * i, 2 * i + 1)] = new vector<DMatch>(lr_mi.matches);
                    cout << "(" << 2 * i << "->" << 2 * i + 1 << ": " << lr_mi.matches.size() << ") ";
                    cout.flush();
                }
                else {
                    const detail::ImageFeatures &features_left = *(features_collection.find(2 * i)->second);
                    vector<Point2f> xy_left;
                    KeyPoint::convert(features_left.keypoints, xy_left);

                    Mat_<float> xy_right, err_vec;
                    vector<uchar> matched;
                    calcOpticalFlowPyrLK(left_imgs[i], right_imgs[i], xy_left, xy_right, matched, err_vec, Size(3, 3), 3);

                    cout << xy_left.size() << " " << xy_right.cols << endl;
                    cin.get();

                    Ptr<detail::ImageFeatures> features_right = new detail::ImageFeatures();
                    Ptr<vector<DMatch> > matches = new vector<DMatch>();

                    for (size_t j = 0; j < matched.size(); ++j) {
                        if (matched[j] && err_vec(0, j) < 5) {
                            matches->push_back(DMatch(j, features_right->keypoints.size(), 0.f));
                            features_right->keypoints.push_back(KeyPoint(xy_right(0, 2 * j), xy_right(0, 2 * j + 1), 0.f));
                        }
                    }

                    features_collection[2 * i + 1] = features_right;
                    matches_collection[make_pair(2 * i, 2 * i + 1)] = matches;

                    cout << "(" << 2 * i << "->" << 2 * i + 1 << ": " << matches->size() << ") ";
                    cout.flush();
                }

                for (int j = i + 1; j < num_frames; ++j) {
                    detail::MatchesInfo ll_mi;
                    (*matcher)(*(features_collection.find(2 * i)->second), *(features_collection.find(2 * j)->second), ll_mi);
                    matches_collection[make_pair(2 * i, 2 * j)] = new vector<DMatch>(ll_mi.matches);
                    cout << "(" << 2 * i << "->" << 2 * j << ": " << ll_mi.matches.size() << ") ";
                    cout.flush();
                }
            }            
        }
        cout << endl;

        if (show_matches) {
            for (int i = 0; i < num_frames; ++i) {
                Mat img;
                drawMatches(left_imgs[i], features_collection.find(2 * i)->second->keypoints, 
                            right_imgs[i], features_collection.find(2 * i + 1)->second->keypoints,
                            *(matches_collection.find(make_pair(2 * i, 2 * i + 1))->second), img);
                Mat img_;
                resize(img, img_, Size(), 0.5, 0.5);
                imshow("matches", img_);
                waitKey();

                for (int j = i + 1; j < num_frames; ++j) {
                    drawMatches(left_imgs[i], features_collection.find(2 * i)->second->keypoints, 
                                left_imgs[j], features_collection.find(2 * j)->second->keypoints,
                                *(matches_collection.find(make_pair(2 * i, 2 * j))->second), img);
                    resize(img, img_, Size(), 0.5, 0.5);
                    imshow("matches", img_);
                    waitKey();
                }
            }
        }

        // Find fundamental matrix and extract camera mat

        cout << "\nFinding F...\n";

        Mat_<double> F = FindFundamentalMatFromPairs(features_collection, matches_collection, F_est_thresh, F_est_conf);


        if (!F_gold.empty()) {
            cout << "F_gold = \n" << F_gold << endl;
            F = F_gold;
        }                        

        cout << "F_final = \n" << F << endl;

        Mat_<double> P_l = Mat::eye(3, 4, CV_64F);
        Mat_<double> P_r = CameraMatFromFundamentalMat(F);

        // Remove outliers, compute confidences

        cout << "\nRemoving outliers...\n";

        RelativeConfidences rel_confs;
        /*ofstream ff("epip_dist_ll.csv");
        ff.close();*/

        for (MatchesCollection::iterator iter = matches_collection.begin();
             iter != matches_collection.end(); ++iter)
        {
            int from = iter->first.first;
            int to = iter->first.second;

            Ptr<vector<DMatch> > matches = iter->second;
            int num_inliers = 0;
            Mat_<uchar> mask;

            if (!matches->empty()) {
                Mat F_;

                if (IsLeftRightPair(from, to)) {
                    F_ = F;
                }
                else if (BothAreLeft(from, to)) {
                    Mat xy1, xy2;
                    ExtractMatchedKeypoints(*(features_collection.find(from)->second),
                                            *(features_collection.find(to)->second),
                                            *matches, xy1, xy2);

                    vector<uchar> mask;
                    F_ = findFundamentalMat(xy1.reshape(2), xy2.reshape(2), mask, FM_LMEDS, F_est_thresh);

                    /*cout << "Inliers rate = " << accumulate(mask.begin(), mask.end(), 0) / (double)mask.size() << endl;

                    ofstream f("epip_dist_ll.csv", ios_base::app);
                    for (size_t i = 0; i < mask.size(); ++i) {
                        if (mask[i]) {
                            double dist = SymEpipDist2(xy2.at<double>(0,2*i), xy2.at<double>(0,2*i+1), F_, xy1.at<double>(0,2*i), xy1.at<double>(0,2*i+1));
                            f << dist << endl;
                        }
                    }
                    f.close();*/
                }
                else {
                    stringstream msg;
                    msg << "from=" << from << ", to=" << to << " - bad matches";
                    throw runtime_error(msg.str());
                }

                num_inliers = FindFundamentalMatInliers(*(features_collection.find(from)->second),
                                                        *(features_collection.find(to)->second),
                                                        *matches, F_, F_est_thresh, mask);
            }

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

            rel_confs[iter->first] = conf;
        }

        // Select confident subset

        set<int> conf_pair_indices;

        for (MatchesCollection::iterator iter = matches_collection.begin();
             iter != matches_collection.end(); ++iter)
        {
            if (IsLeftRightPair(iter->first.first, iter->first.second) && rel_confs[iter->first] > conf_thresh) {
                conf_pair_indices.insert(iter->first.first / 2);
            }
        }

        MatchesCollection conf_matches_collection;
        RelativeConfidences good_rel_confs;

        for (MatchesCollection::iterator iter = matches_collection.begin();
             iter != matches_collection.end(); ++iter)
        {
            bool is_conf_lr_pair = IsLeftRightPair(iter->first.first, iter->first.second)
                                   && rel_confs[iter->first] > conf_thresh
                                   && conf_pair_indices.find(iter->first.first / 2) != conf_pair_indices.end();

            bool is_conf_ll_pair = BothAreLeft(iter->first.first, iter->first.second)
                                   && rel_confs[iter->first] > conf_thresh;

            if (is_conf_ll_pair || is_conf_lr_pair) {
                conf_matches_collection[iter->first] = iter->second;
                good_rel_confs[iter->first] = rel_confs[iter->first];
            }
        }

        // Affine rectification

        map<pair<int, int>, Mat> Ps_l_a;
        map<pair<int, int>, Mat> Ps_r_a;
        HomographiesP2 Hs_inf;
        HomographiesP3 Hs_01_a;

        for (MatchesCollection::iterator iter = conf_matches_collection.begin();
             iter != conf_matches_collection.end(); ++iter)
        {
            if (BothAreLeft(iter->first.first, iter->first.second)) {
                cout << "\nPROCESSING MATCH " << iter->first.first << "->" << iter->first.second << endl;

                // Get image indices
                int from = iter->first.first / 2;
                int to = iter->first.second / 2;

                Ptr<vector<DMatch> > matches_lr0 = matches_collection.find(make_pair(2 * from, 2 * from + 1))->second;
                Ptr<vector<DMatch> > matches_lr1 = matches_collection.find(make_pair(2 * to, 2 * to + 1))->second;
                Ptr<vector<DMatch> > matches_ll = matches_collection.find(make_pair(2 * from, 2 * to))->second;

                Mat_<double> xy_l0, xy_r0, xy_l1, xy_r1;
                ExtractMatchedKeypoints(*(features_collection.find(2 * from)->second),
                                        *(features_collection.find(2 * from + 1)->second), *matches_lr0, xy_l0, xy_r0);
                ExtractMatchedKeypoints(*(features_collection.find(2 * to)->second),
                                        *(features_collection.find(2 * to + 1)->second), *matches_lr1, xy_l1, xy_r1);

                Mat_<double> Hpa, H01_a;
                Mat_<double> xyzw0_a, xyzw1_a;
                Mat_<double> P_l_a_ = P_l.clone();
                Mat_<double> P_r_a_ = P_r.clone();

                AffineRectifyStereoCameraByTwoShots(P_l_a_, P_r_a_, xy_l0, xy_r0, xy_l1, xy_r1, matches_lr0, matches_lr1, matches_ll,
                                                    H_est_num_iters, H_est_subset_size, H_est_thresh,
                                                    Hpa, H01_a, xyzw0_a, xyzw1_a);

                Hs_01_a[make_pair(from, to)] = H01_a;

                Ps_l_a[make_pair(from, to)] = P_l_a_;
                Ps_r_a[make_pair(from, to)] = P_r_a_;

                // Stereo pair relative rotation can be very close to the identity matrix. That
                // can lead to numerical instability in K estimation process, so we avoid using those
                // rotations in the linear autocalibration algorithm.

                Hs_inf[make_pair(2 * from, 2 * to)] = Mat(P_l_a_ * H01_a.inv())(Rect(0, 0, 3, 3));
                //Hs_inf[make_pair(2 * from, 2 * from + 1)] = P_r_a_(Rect(0, 0, 3, 3));
            }
        }

        Mat_<double> K_est, R_est, T_est;

        // Linear autocalibration

        if (K_init.empty()) {
            if (!K1_gold.empty())
                K_init = K1_gold;
        }
        else {
            K_init(0,0) *= work_size.width / (double)left_imgs[0].cols;
            K_init(0,1) *= work_size.width / (double)left_imgs[0].cols;
            K_init(0,2) *= work_size.width / (double)left_imgs[0].cols;
            K_init(1,1) *= work_size.height / (double)left_imgs[0].rows;
            K_init(1,2) *= work_size.height / (double)left_imgs[0].rows;
        }

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

            SVD svd(R01, SVD::FULL_UV);
            R01 = svd.u * svd.vt;
            if (determinant(R01) < 0)
                R01 *= -1;

            rel_motions[iter->first] = Motion(R01, T01);           

            Mat rvec01;
            Rodrigues(R01, rvec01);

            RigidCamera rigid_cam = RigidCamera::FromProjectiveMat(Ps_r_a[iter->first] * Ham);

            Mat rvec;
            Rodrigues(rigid_cam.R(), rvec);
            total_rvec += rvec;
            cout << "(" << iter->first.first << "->" << iter->first.second << "): R=" << rvec
                 << ", T=" << rigid_cam.T() / rigid_cam.T().at<double>(0, 0)
                 << ", conf=" << good_rel_confs.find(make_pair(iter->first.first * 2, iter->first.second * 2))->second
                 << endl;

            total_T += rigid_cam.T();
            total_estimations++;
        }

        Mat avg_R;
        Rodrigues(total_rvec / total_estimations, avg_R);

        Mat avg_T = total_T / total_estimations;

        detail::Graph eff_corresp;
        RelativeConfidences ll_rel_confs;
        for (RelativeConfidences::iterator iter = good_rel_confs.begin(); iter != good_rel_confs.end(); ++iter) {
            if (BothAreLeft(iter->first.first, iter->first.second)) {
                ll_rel_confs[make_pair(iter->first.first / 2, iter->first.second / 2)] = iter->second;
            }
        }
        int ref_pair_idx = ExtractEfficientCorrespondences(num_frames, ll_rel_confs, eff_corresp);

        AbsoluteMotions abs_motions;
        CalcAbsoluteMotions(rel_motions, eff_corresp, ref_pair_idx, abs_motions);

        Mat_<double> K_norm = K_init.inv();
        for (FeaturesCollection::iterator iter = features_collection.begin(); iter != features_collection.end(); ++iter) {
            detail::ImageFeatures &f = *(iter->second);
            for (size_t i = 0; i < f.keypoints.size(); ++i) {
                Point2f &kp = f.keypoints[i].pt;
                double x = K_norm(0, 0) * kp.x + K_norm(0, 1) * kp.y + K_norm(0, 2);
                double y = K_norm(1, 1) * kp.y + K_norm(1, 2);
                kp.x = static_cast<float>(x);
                kp.y = static_cast<float>(y);
            }
        }

        RigidCamera P_r_m(K_norm * K_init, avg_R.clone(), avg_T.clone());

        double final_rms_error = RefineStereoCamera(P_r_m, abs_motions, features_collection, matches_collection, ~REFINE_FLAG_K_SKEW);
        final_rms_error = RefineStereoCamera(P_r_m, abs_motions, features_collection, matches_collection, ~REFINE_FLAG_K_SKEW);
        final_rms_error = RefineStereoCamera(P_r_m, abs_motions, features_collection, matches_collection, ~REFINE_FLAG_K_SKEW);

        P_r_m = RigidCamera(K_norm.inv() * P_r_m.K(), P_r_m.R(), P_r_m.T());
        cout << "\nK_refined = \n" << P_r_m.K() << endl;

        cout << "\nSUMMARY\n";

        Mat R_, T_;
        Mat E = P_r_m.K().t() * F * P_r_m.K();
        DecomposeEssentialMat(E, R_, T_);
        Mat tmp;
        Rodrigues(R_, tmp);
        cout << "R(E) = " << tmp << endl
             << "T(E) = " << T_ / T_.at<double>(0, 0) << endl;

        K_est = P_r_m.K();
        R_est = P_r_m.R().t();
        T_est = -P_r_m.R().t() * P_r_m.T();
        T_est /= T_est(0, 0);

        Mat_<double> rvec_est;
        Rodrigues(R_est, rvec_est);
        cout << "rvec_est = " << rvec_est << endl;

        if (!R_gold.empty()) {
            Mat_<double> rvec_gold;
            Rodrigues(R_gold, rvec_gold);
            cout << "rvec_gold = " << rvec_gold << endl;
        }

        cout << "T_est = " << T_est << endl;

        if (!T_gold.empty())
            cout << "T_gold = " << T_gold / T_gold(0, 0) << endl;        

        cout << "K_est = \n" << K_est << endl;

        if (!log_file.empty()) {
            ofstream f(log_file.c_str(), ios_base::app);
            f << K_init(0, 0) << ";" << K_init(1, 1) << ";" << K_init(0, 2) << ";" << K_init(1, 2) << ";" << K_init(0, 1) << ";"
              << K_est(0, 0) << ";" << K_est(1, 1) << ";" << K_est(0, 2) << ";" << K_est(1, 2) << ";" << K_est(0, 1) << ";"
              << rvec_est(0, 0) << ";" << rvec_est(0, 1) << ";" << rvec_est(0, 2) << ";"
              << T_est(0, 0) << ";" << T_est(1, 0) << ";" << T_est(2, 0) << ";"
              << final_rms_error << ";" << num_frames << ";";
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
        else if (string(argv[i]) == "--blur-ksize")
            blur_ksize = atoi(argv[++i]);
        else if (string(argv[i]) == "--manual-registr") 
            manual_registr = atoi(argv[++i]);
        else if (string(argv[i]) == "--save-keypoints")
            save_keypoints = atoi(argv[++i]);
        else if (string(argv[i]) == "--load-keypoints")
            load_keypoints = atoi(argv[++i]);
        else if (string(argv[i]) == "--save-matches")
            save_matches = atoi(argv[++i]);
        else if (string(argv[i]) == "--load-matches")
            load_matches = atoi(argv[++i]);
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
            else if (string(argv[i + 1]) == "opt_flow")
                opt_flow_matching = true;
            else if (string(argv[i + 1]) == "opt_assign")
                opt_assignment_matching = true;
            else
                throw runtime_error(string("Unknown matcher type: ") + argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--show-matches")
            show_matches = static_cast<bool>(atoi(argv[++i]));
        else if (string(argv[i]) == "--match-conf")
            features_matcher_creator.match_conf = static_cast<float>(atof(argv[++i]));
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
            H_est_num_iters = atoi(argv[++i]);
        else if (string(argv[i]) == "--H-est-subset-size")
            H_est_subset_size = atoi(argv[++i]);
        else if (string(argv[i]) == "--H-est-thresh")
            H_est_thresh = atof(argv[++i]);
        else if (string(argv[i]) == "--conf-thresh")
            conf_thresh = atof(argv[++i]);
        else if (string(argv[i]) == "--log-file")
            log_file = argv[++i];
        else if (string(argv[i]) == "--intrinsics-file")
            intrinsics_file = argv[++i];
        else if (string(argv[i]) == "--extrinsics-file")
            extrinsics_file = argv[++i];
        else if (string(argv[i]) == "--work-size") {
            work_size.width = atoi(argv[i + 1]);
            work_size.height = atoi(argv[i + 2]);
            i += 2;
        }
        else {
            if (i < argc - 1) {
                img_names.push_back(make_pair(argv[i], argv[i + 1]));
                i++;
            }
            else
                throw runtime_error("Can't find right camera image");
        }
    }
}

