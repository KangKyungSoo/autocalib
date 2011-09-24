#include <include/evaluation.h>

using namespace std;
using namespace cv;

namespace autocalib {

Ptr<detail::ImageFeatures> SyntheticScene::TakeShot(const RigidCamera &camera, Rect roi,
                                                    Mat *img)
{
    Mat R_inv = camera.R().inv();
    Point3d origin = Mat(-R_inv * camera.T()).at<Point3d>(0, 0);

    vector<int> visible_points;
    Ptr<detail::ImageFeatures> features = new detail::ImageFeatures;

    Mat_<double> P = camera.P();
    for (size_t i = 0; i < points.size(); ++i) {
        Point3d pt = points[i];
        if (IsVisible(pt, origin)) {
            double x = P(0, 0) * pt.x + P(0, 1) * pt.y + P(0, 2) * pt.z + P(0, 3);
            double y = P(1, 0) * pt.x + P(1, 1) * pt.y + P(1, 2) * pt.z + P(1, 3);
            double z = P(2, 0) * pt.x + P(2, 1) * pt.y + P(2, 2) * pt.z + P(2, 3);
            Point2f kp(float(x / z), float(y / z));
            if (kp.x > (float)roi.x && kp.x < float(roi.width - 1) &&
                kp.y > (float)roi.y && kp.y < float(roi.height - 1))
            {
                visible_points.push_back(i);
                features->keypoints.push_back(KeyPoint(kp, 1.f));
            }
        }
    }

    features->descriptors.create(visible_points.size(), 1, CV_32S);
    for (size_t i = 0; i < visible_points.size(); ++i)
        features->descriptors.at<int>(i ,0) = visible_points[i];

    features->img_size = roi.size();

    if (img) {
        img->create(roi.size(), CV_8U);
        img->setTo(0);
        for (size_t i = 0; i < visible_points.size(); ++i)
            circle(*img, features->keypoints[i].pt, 1, Scalar::all(255), 2);
    }

    return features;
}

} // namespace autocalib
