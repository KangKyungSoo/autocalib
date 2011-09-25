#include <iostream>
#include <string>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <core/include/core.h>
#include <evaluation/include/evaluation.h>

using namespace std;
using namespace cv;
using namespace autocalib;

int num_points = 1000;
int num_cameras = 5;
Rect viewport = Rect(0, 0, 1920, 1080);
Mat_<double> K_gold;
Mat_<double> camera_center;
double max_angle = 0.1;
bool create_images = true;

int main(int argc, char **argv) {    
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--num-points")
            num_points = atoi(argv[++i]);
        else if (string(argv[i]) == "--num-cameras")
            num_cameras = atoi(argv[++i]);
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
        }
        else if (string(argv[i]) == "--camera-center") {
            camera_center = Mat::zeros(3, 1, CV_64F);
            camera_center(0, 0) = atof(argv[i + 1]);
            camera_center(1, 0) = atof(argv[i + 2]);
            camera_center(2, 0) = atof(argv[i + 3]);
            i += 3;
        }
        else if (string(argv[i]) == "--max-angle")
            max_angle = atof(argv[++i]);
        else if (string(argv[i]) == "--create-images")
            create_images = atoi(argv[++i]);
        else {
            cout << "Can't parse CLI args\n";
            return -1;
        }
    }
    if (K_gold.empty()) {
        K_gold = Mat::eye(3, 3, CV_64F);
        K_gold(0, 0) = K_gold(1, 1) = viewport.width + viewport.height;
        K_gold(0, 2) = viewport.width * 0.5;
        K_gold(1, 2) = viewport.height * 0.5;
    }
    if (camera_center.empty()) {
        camera_center = Mat::zeros(3, 1, CV_64F);
        camera_center(2, 0) = -10;
    }

    Ptr<SyntheticScene> scene = new SphereScene(num_points);
    vector<Ptr<detail::ImageFeatures> > features(num_cameras);

    for (int i = 0; i < num_cameras; ++i) {
        Mat rvec = Mat::zeros(3, 1, CV_64F);
        randu(rvec, -1, 1);
        rvec /= norm(rvec) / (rand() / (double)RAND_MAX * max_angle);

        Mat R;
        Rodrigues(rvec, R);
        RigidCamera camera = RigidCamera::LocalToWorld(K_gold, R, camera_center);

        if (!create_images)
            features[i] = scene->TakeShot(camera, viewport);
        else {
            Mat image;
            features[i] = scene->TakeShot(camera, viewport, &image);
            stringstream name;
            name << "camera" << i << ".jpg";
            imwrite(name.str(), image);
        }
    }

    return 0;
}
