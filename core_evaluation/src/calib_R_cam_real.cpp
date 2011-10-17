#include <iostream>
#include <vector>
#include <set>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <core/include/core.h>

using namespace std;
using namespace cv;
using namespace autocalib;


// TODO does flann matches work with ORB (uchar) descriptors?
/** Describes an ORB features finder. */
class OrbFeaturesFinder : public detail::FeaturesFinder {
public:

    /** Constructs ORB features finder.
      *
      * \param num_features Number of desired features
      */
    OrbFeaturesFinder(int num_features) : orb_(num_features) {}

private:
    virtual void find(const Mat &image, detail::ImageFeatures &features) {
        Mat tmp;
        orb_(image, Mat(), features.keypoints, tmp);
        tmp.convertTo(features.descriptors, CV_32F);
    }

    ORB orb_;
};


/** Base class for features finder creators */
class FeaturesFinderCreator {
public:
    virtual ~FeaturesFinderCreator() {}

    /** Creates a features finder.
      *
      * \return Pointer to features finder object
      */
    virtual Ptr<detail::FeaturesFinder> Create() = 0;
};


class SurfFeaturesFinderCreator : public FeaturesFinderCreator {
public:
    SurfFeaturesFinderCreator() : hess_thresh(300), num_octaves(3), num_layers(4) {}

    virtual Ptr<detail::FeaturesFinder> Create() {
        return new detail::SurfFeaturesFinder(hess_thresh, num_octaves, num_layers);
    }

    double hess_thresh;
    int num_octaves;
    int num_layers;
};


class OrbFeaturesFinderCreator : public FeaturesFinderCreator {
public:
    OrbFeaturesFinderCreator() : num_features(500) {}

    virtual Ptr<detail::FeaturesFinder> Create() {
        return new OrbFeaturesFinder(num_features);
    }

    int num_features;
};


class BestOf2NearestMatcher : public detail::FeaturesMatcher {
public:
    BestOf2NearestMatcher(Ptr<DescriptorMatcher> &matcher, float match_conf)
        : matcher_(matcher), match_conf_(match_conf) {}

    virtual void match(const detail::ImageFeatures &f1, const detail::ImageFeatures &f2,
                       detail::MatchesInfo &mi)
    {
        vector<vector<DMatch> > matches;
        set<pair<int, int> > matches12;

        matcher_->knnMatch(f1.descriptors, f2.descriptors, matches, 2);
        for (size_t i = 0; i < matches.size(); ++i) {
            if (matches[i].size() < 2)
                continue;
            const DMatch &m1 = matches[i][0];
            const DMatch &m2 = matches[i][1];
            if (m1.distance < (1.f - match_conf_) * m2.distance)
                matches12.insert(make_pair(m1.queryIdx, m1.trainIdx));
        }

        mi.matches.clear();
        matcher_->knnMatch(f2.descriptors, f1.descriptors, matches, 2);
        for (size_t i = 0; i < matches.size(); ++i) {
            if (matches[i].size() < 2)
                continue;
            const DMatch &m1 = matches[i][0];
            const DMatch &m2 = matches[i][1];
            if (m1.distance < (1.f - match_conf_) * m2.distance &&
                matches12.find(make_pair(m1.trainIdx, m1.queryIdx)) != matches12.end())
            {
                mi.matches.push_back(DMatch(m1.trainIdx, m1.queryIdx, m1.distance));
            }
        }
    }

private:
    Ptr<DescriptorMatcher> matcher_;
    float match_conf_;
};


class BestOf2NearestMatcherCreator {
public:
    BestOf2NearestMatcherCreator()
        : matcher(new FlannBasedMatcher()), match_conf(0.65f) {}

    Ptr<detail::FeaturesMatcher> Create() {
        return new BestOf2NearestMatcher(matcher, match_conf);
    }

    Ptr<DescriptorMatcher> matcher;
    float match_conf;
};


void ParseArgs(int argc, char **argv);

vector<string> img_names;
vector<Mat> imgs;
Ptr<FeaturesFinderCreator> features_finder_creator = new SurfFeaturesFinderCreator();
BestOf2NearestMatcherCreator matcher_creator;
FeaturesCollection features_collection;
int min_num_matches = 6;
double H_est_thresh = 3.;
Mat_<double> K_guess;
Mat_<double> K_init;
bool lin_est_skew = false;
bool refine_skew = false;

int main(int argc, char **argv) {
    try {
        ParseArgs(argc, argv);

        if (K_guess.empty()) {
            K_guess = Mat::eye(3, 3, CV_64F);
            K_guess(0, 0) = imgs[0].cols + imgs[0].rows; K_guess(0, 2) = imgs[0].cols * 0.5;
            K_guess(1, 1) = imgs[0].cols + imgs[0].rows; K_guess(1, 2) = imgs[0].rows * 0.5;
        }
        cout << "K_guess =\n" << K_guess << endl;

        int num_cameras = static_cast<int>(imgs.size());

        // Find features

        cout << "Finding features...\n";
        Ptr<detail::FeaturesFinder> features_finder = features_finder_creator->Create();
        features_collection.resize(imgs.size());

        for (int i = 0; i < num_cameras; ++i) {
            int64 t = getTickCount();
            cout << "Finding features in '" << img_names[i] << "'... ";
            (*features_finder)(imgs[i], features_collection[i]);
            cout << "#features = " << features_collection[i].keypoints.size()
                 << ", time = " << (getTickCount() - t) / getTickFrequency() << " sec\n";
        }

        // Match all pairs

        cout << "Matching pairs...\n";
        Ptr<detail::FeaturesMatcher> matcher = matcher_creator.Create();
        vector<detail::MatchesInfo> pairwise_matches;
        (*matcher)(features_collection, pairwise_matches);

        // Convert pairwise matches into our matches format

        MatchesCollection matches_collection;

        for (int from = 0; from < num_cameras - 1; ++from) {
            for (int to = from + 1; to < num_cameras; ++to) {
                const detail::MatchesInfo &mi = pairwise_matches[from * imgs.size() + to];
                matches_collection.insert(make_pair(make_pair(from, to), mi.matches));
            }
        }

        // Estimate homographies

        vector<Mat> Hs;
        vector<Mat> Hs_from_0;
        Mat keypoints1, keypoints2;

        cout << "Estimating Hs...\n";
        for (int from = 0; from < num_cameras - 1; ++from) {
            for (int to = from + 1; to < num_cameras; ++to) {
                const vector<DMatch> &matches = matches_collection.find(make_pair(from, to))->second;

                cout << "Estimating H between '" << img_names[from] << "' and '" << img_names[to]
                     << "'... #matches = " << matches.size();

                if (matches.size() < min_num_matches) {
                    cout << ", not enough matches\n";
                    continue;
                }

                ExtractMatchedKeypoints(features_collection[from], features_collection[to],
                                        matches, keypoints1, keypoints2);
                vector<uchar> inliers_mask;
                Mat H = findHomography(keypoints1, keypoints2, inliers_mask, RANSAC, H_est_thresh);

                if (H.empty()) {
                    cout << ", can't estimate H\n";
                    continue;
                }

                int num_inliers = 0;
                for (size_t i = 0; i < matches.size(); ++i)
                    if (inliers_mask[i])
                        num_inliers++;
                cout << ", #inliers = " << num_inliers << endl;
                Hs.push_back(H);
                if (from == 0)
                    Hs_from_0.push_back(H);
            }
        }

        // Linear calibration

        if (K_init.empty()) {
            cout << "Linear calibrating...\n";
            if (lin_est_skew)
                K_init = CalibRotationalCameraLinear(Hs, K_guess);
            else
                K_init = CalibRotationalCameraLinearNoSkew(Hs, K_guess);
            cout << "K_init =\n" << K_init << endl;
        }

        // Non-linear refinement

        cout << "Refining camera...\n";

        if (Hs_from_0.size() != num_cameras - 1) {
            stringstream msg;
            msg << "Refinement requires Hs between first and all other images, "
                << "but only " << Hs_from_0.size() << " were/was found";
            throw runtime_error(msg.str());
        }

        vector<Mat> Rs(num_cameras);
        Rs[0] = Mat::eye(3, 3, CV_64F);
        for (int i = 1; i < num_cameras; ++i)
            Rs[i] = K_init.inv() * Hs_from_0[i - 1] * K_init;

        Mat_<double> K_refined = K_init.clone();
        if (refine_skew)
            RefineRigidCamera(K_refined, Rs, features_collection, matches_collection);
        else {
            K_refined(0, 1) = 0;
            RefineRigidCamera(K_refined, Rs, features_collection, matches_collection,
                              REFINE_FLAG_ALL & ~REFINE_FLAG_SKEW);
        }
        cout << "K_refined =\n" << K_refined << endl;

        cout << "SUMMARY\n";
        cout << "K_guess =\n" << K_guess << endl;
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
        else if (string(argv[i]) == "--matcher") {
            if (string(argv[i + 1]) == "bfm_l2")
                matcher_creator.matcher = new BruteForceMatcher<L2<float> >();
            else if (string(argv[i + 1]) == "flann")
                matcher_creator.matcher = new FlannBasedMatcher();
            else
                throw runtime_error(string("Unknown matcher type: ") + argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--match-conf")
            matcher_creator.match_conf = atof(argv[++i]);
        else if (string(argv[i]) == "--min-num-matches")
            min_num_matches = atoi(argv[++i]);
        else if (string(argv[i]) == "--H-est-thresh")
            H_est_thresh = atof(argv[++i]);
        else if (string(argv[i]) == "--K-guess") {
            K_guess = Mat::eye(3, 3, CV_64F);
            K_guess(0, 0) = atof(argv[i + 1]);
            K_guess(0, 1) = atof(argv[i + 2]);
            K_guess(0, 2) = atof(argv[i + 3]);
            K_guess(1, 1) = atof(argv[i + 4]);
            K_guess(1, 2) = atof(argv[i + 5]);
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
        else {
            Mat img = imread(argv[i]);
            if (img.empty())
                throw runtime_error(string("Can't open image: ") + argv[i]);
            img_names.push_back(argv[i]);
            imgs.push_back(img);
        }
    }
}
