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

vector<Mat> imgs;
Ptr<FeaturesFinderCreator> finder_creator = new SurfFeaturesFinderCreator();

int main(int argc, char **argv) {
    try {
        ParseArgs(argc, argv);
    }
    catch (const exception &e) {
        cout << "Error: " << e.what() << endl;
    }
    return 0;
}


void ParseArgs(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        Mat img = imread(argv[++i]);
        if (img.empty())
            throw runtime_error(string("Can't open image: ") + argv[i]);
    }
}
