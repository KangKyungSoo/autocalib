#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
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
bool create_images = false;
double H_est_thresh = 3;
bool add_noise = false;
double noise_stddev = 1;

int main(int argc, char **argv) {    
    try {
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
                i += 5;
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
            else if (string(argv[i]) == "--H-est-thresh")
                H_est_thresh = atof(argv[++i]);
            else if (string(argv[i]) == "--add-noise")
                add_noise = atoi(argv[++i]);
            else if (string(argv[i]) == "--noise-stddev")
                noise_stddev = atof(argv[++i]);
            else
                throw runtime_error(string("Can't parse command line arg: ") + argv[i]);
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
        vector<RigidCamera> cameras(num_cameras);
        vector<Ptr<detail::ImageFeatures> > features(num_cameras);

        // Generate cameras and shots
        for (int i = 0; i < num_cameras; ++i) {
            Mat rvec = Mat::zeros(3, 1, CV_64F);
            randu(rvec, -1, 1);
            rvec /= norm(rvec) / (rand() / (double)RAND_MAX * max_angle);

            Mat R;
            Rodrigues(rvec, R);
            cameras[i] = RigidCamera::LocalToWorld(K_gold, R, camera_center);

            if (!create_images)
                features[i] = scene->TakeShot(cameras[i], viewport);
            else {
                Mat image;
                features[i] = scene->TakeShot(cameras[i], viewport, &image);
                stringstream name;
                name << "camera" << i << ".jpg";
                imwrite(name.str(), image);
            }
        }

        if (add_noise) {
            cout << "Adding noise...\n";
            for (int i = 0; i < num_cameras; ++i) {
                Mat_<float> noise(1, 2 * features[i]->keypoints.size());
                randn(noise, 0, noise_stddev);
                double total_noise = 0;
                for (size_t j = 0; j < features[i]->keypoints.size(); ++j) {
                    features[i]->keypoints[j].pt.x += noise(0, 2 * j);
                    features[i]->keypoints[j].pt.y += noise(0, 2 * j + 1);
                    total_noise += noise(0, 2 * j) * noise(0, 2 * j) + noise(0, 2 * j + 1) * noise(0, 2 * j + 1);
                }
                cout << "Shot " << i << " noise RMS: " << sqrt(total_noise / features[i]->keypoints.size()) << endl;
            }
        }

        cout << "Finding homographies...\n";
        vector<Mat> Hs;
        Mat kps1, kps2;
        vector<DMatch> matches;
        for (int i = 0; i < num_cameras - 1; ++i) {
            for (int j = i + 1; j < num_cameras; ++j) {
                MatchSyntheticShots((*features[i]), (*features[j]), matches);
                ExtractMatchedKeypoints((*features[i]), (*features[j]), matches, kps1, kps2);

                Mat_<double> H = findHomography(kps1, kps2, cv::RANSAC, H_est_thresh);
                if (H.empty())
                    cout << "Can't find H from " << i << " to " << j << endl;
                else {
                    Hs.push_back(H);
                    double err = 0;
                    for (size_t k = 0; k < matches.size(); ++k) {
                        Point2f kp1 = kps1.at<Point2f>(0, k);
                        Point2f kp2 = kps2.at<Point2f>(0, k);
                        double x = H(0, 0) * kp1.x + H(0, 1) * kp1.y + H(0, 2);
                        double y = H(1, 0) * kp1.x + H(1, 1) * kp1.y + H(1, 2);
                        double z = H(2, 0) * kp1.x + H(2, 1) * kp1.y + H(2, 2);
                        err += (kp2.x - x / z) * (kp2.x - x / z) + (kp2.y - y / z) * (kp2.y - y / z);
                    }
                    cout << "H from " << i << " to " << j << " RMS error: " << sqrt(err / matches.size()) << endl;
                }
            }
        }

        cout << "Linear calibrating...\n";
        Mat_<double> K_final = CalibRotationalCameraLinear(Hs);

        cout << "K_gold:\n" << K_gold << endl;
        cout << "K_final:\n" << K_final << endl;
    }
    catch (const exception &e) {
        cout << "Error: " << e.what() << "\n";
    }

    return 0;
}
