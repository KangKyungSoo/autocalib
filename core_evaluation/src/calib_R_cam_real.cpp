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


void ParseArgs(int argc, char **argv);

vector<string> img_names;
vector<Mat> imgs;
Ptr<FeaturesFinderCreator> ffinder_creator = new SurfFeaturesFinderCreator();
vector<detail::ImageFeatures> features;

int main(int argc, char **argv) {
    try {
        ParseArgs(argc, argv);

        // Find features

        Ptr<detail::FeaturesFinder> ffinder = ffinder_creator->Create();
        features.resize(imgs.size());

        for (size_t i = 0; i < imgs.size(); ++i) {
            int64 t = getTickCount();
            cout << "Finding features in image " << img_names[i] << "... ";
            (*ffinder)(imgs[i], features[i]);
            cout << "#features = " << features[i].keypoints.size() << ", "
                 << (getTickCount() - t) / getTickFrequency() << " sec\n";
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
                ffinder_creator = new SurfFeaturesFinderCreator();
            else if (string(argv[i + 1]) == "orb")
                ffinder_creator = new OrbFeaturesFinderCreator();
            else
                throw runtime_error(string("Unknown features finder type: ") + argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--surf-hess-thresh") {
            FeaturesFinderCreator *ffc = static_cast<FeaturesFinderCreator*>(ffinder_creator);
            SurfFeaturesFinderCreator *sffc = dynamic_cast<SurfFeaturesFinderCreator*>(ffc);
            if (!sffc)
                throw runtime_error(string("Inconsistent features finder option: ") + argv[i + 1]);
            sffc->hess_thresh = atoi(argv[++i]);
        }
        else if (string(argv[i]) == "--surf-num-octaves") {
            FeaturesFinderCreator *ffc = static_cast<FeaturesFinderCreator*>(ffinder_creator);
            SurfFeaturesFinderCreator *sffc = dynamic_cast<SurfFeaturesFinderCreator*>(ffc);
            if (!sffc)
                throw runtime_error(string("Inconsistent features finder option: ") + argv[i + 1]);
            sffc->num_octaves = atoi(argv[++i]);
        }
        else if (string(argv[i]) == "--surf-num-layers") {
            FeaturesFinderCreator *ffc = static_cast<FeaturesFinderCreator*>(ffinder_creator);
            SurfFeaturesFinderCreator *sffc = dynamic_cast<SurfFeaturesFinderCreator*>(ffc);
            if (!sffc)
                throw runtime_error(string("Inconsistent features finder option: ") + argv[i + 1]);
            sffc->num_layers = atoi(argv[++i]);
        }
        else if (string(argv[i]) == "--orb-num-features") {
            FeaturesFinderCreator *ffc = static_cast<FeaturesFinderCreator*>(ffinder_creator);
            OrbFeaturesFinderCreator *offc = dynamic_cast<OrbFeaturesFinderCreator*>(ffc);
            if (!offc)
                throw runtime_error(string("Inconsistent features finder option: ") + argv[i + 1]);
            offc->num_features = atoi(argv[++i]);
        }
        else {
            Mat img = imread(argv[++i]);
            if (img.empty())
                throw runtime_error(string("Can't open image: ") + argv[i]);
            img_names.push_back(argv[i]);
            imgs.push_back(img);
        }
    }
}
