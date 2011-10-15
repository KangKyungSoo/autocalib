#include <iostream>
#include <vector>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <core/include/core.h>

using namespace std;
using namespace cv;
using namespace autocalib;


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
        orb_(image, Mat(), features.keypoints, features.descriptors);
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


class BestOf2NearestMatcherCreator {
public:
    BestOf2NearestMatcherCreator() : match_conf(0.65f) {}

    Ptr<detail::FeaturesMatcher> Create() {
        return new detail::BestOf2NearestMatcher(false, match_conf);
    }

    float match_conf;
};


void ParseArgs(int argc, char **argv);

vector<string> img_names;
vector<Mat> imgs;
Ptr<FeaturesFinderCreator> features_finder_creator = new SurfFeaturesFinderCreator();
BestOf2NearestMatcherCreator matcher_creator;
FeaturesCollection features_collection;
double H_est_thresh = 3.;

int main(int argc, char **argv) {
    try {
        ParseArgs(argc, argv);

        // Find features

        Ptr<detail::FeaturesFinder> features_finder = features_finder_creator->Create();
        features_collection.resize(imgs.size());

        for (size_t i = 0; i < imgs.size(); ++i) {
            int64 t = getTickCount();
            cout << "Finding features in image " << img_names[i] << "... ";
            (*features_finder)(imgs[i], features_collection[i]);
            cout << "#features = " << features_collection[i].keypoints.size() << ", "
                 << (getTickCount() - t) / getTickFrequency() << " sec\n";
        }

        // Match all pairs

        Ptr<detail::FeaturesMatcher> matcher = matcher_creator.Create();
        vector<detail::MatchesInfo> pairwise_matches;
        (*matcher)(features_collection, pairwise_matches);

        // Convert pairwise matches into our matches format

        MatchesCollection matches_collection;

        for (size_t from = 0; from + 1 < imgs.size(); ++from) {
            for (size_t to = from + 1; to < imgs.size(); ++to) {
                const detail::MatchesInfo &mi = pairwise_matches[from * imgs.size() + to];
                matches_collection.insert(make_pair(make_pair(from, to), mi.matches));
            }
        }

        // Estimate homographies

        vector<Mat> Hs;
        Mat keypoints1, keypoints2;

        for (size_t from = 0; from + 1 < imgs.size(); ++from) {
            for (size_t to = from + 1; to < imgs.size(); ++to) {
                const vector<DMatch> &matches = matches_collection.find(make_pair(from, to))->second;
                ExtractMatchedKeypoints(features_collection[from], features_collection[to],
                                        matches, keypoints1, keypoints2);

                cout << "Estimating H between #" << from << " and #" << to << "... ";
                vector<uchar> inliers_mask;
                Mat H = findHomography(keypoints1, keypoints2, inliers_mask, RANSAC, H_est_thresh);

                if (H.empty())
                    cout << "FAILED\n";
                else {
                    int num_inliers = 0;
                    for (size_t i = 0; i < matches.size(); ++i)
                        if (inliers_mask[i])
                            num_inliers++;
                    cout << "#matches = " << matches.size() << " #inliers = " << num_inliers << endl;
                    Hs.push_back(H);
                }
            }
        }
    }
    catch (const exception &e) {
        cout << "Error: " << e.what() << endl;
    }
    return 0;
}


void ParseArgs(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--ffinder") {
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
        else if (string(argv[i]) == "--match-conf")
            matcher_creator.match_conf = atof(argv[++i]);
        else if (string(argv[i]) == "--H-est-thresh")
            H_est_thresh = atof(argv[++i]);
        else {
            Mat img = imread(argv[++i]);
            if (img.empty())
                throw runtime_error(string("Can't open image: ") + argv[i]);
            img_names.push_back(argv[i]);
            imgs.push_back(img);
        }
    }
}
