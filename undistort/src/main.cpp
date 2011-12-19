#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void ParseCmdArgs(int argc, char **argv);

vector<string> img_names;
Mat_<double> K_left, K_right;
Mat_<double> dist_left, dist_right;

int main(int argc, char **argv) {
    try {
        ParseCmdArgs(argc, argv);

        Mat left, right;
        Mat left_undist, right_undist;

        for (size_t i = 0; i < img_names.size(); i += 2) {
            left = imread(img_names[i]);
            CV_Assert(!left.empty());
            right = imread(img_names[i + 1]);
            CV_Assert(!right.empty());

            undistort(left, left_undist, K_left, dist_left);
            undistort(right, right_undist, K_right, dist_right);

            stringstream ss;
            ss << "undist_" << img_names[i];
            imwrite(ss.str(), left_undist);

            ss.str("");
            ss << "undist_" << img_names[i + 1];
            imwrite(ss.str(), right_undist);
        }
    }
    catch (const exception &e) {
        cout << "Error: " << e.what() << endl;
    }
    return 0;
}


void ParseCmdArgs(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--K-left") {
            K_left = Mat::eye(3, 3, CV_64F);
            K_left(0, 0) = atof(argv[i + 1]);
            K_left(0, 1) = atof(argv[i + 2]);
            K_left(0, 2) = atof(argv[i + 3]);
            K_left(1, 1) = atof(argv[i + 4]);
            K_left(1, 2) = atof(argv[i + 5]);
            i += 5;
        }
        else if (string(argv[i]) == "--K-right") {
            K_right = Mat::eye(3, 3, CV_64F);
            K_right(0, 0) = atof(argv[i + 1]);
            K_right(0, 1) = atof(argv[i + 2]);
            K_right(0, 2) = atof(argv[i + 3]);
            K_right(1, 1) = atof(argv[i + 4]);
            K_right(1, 2) = atof(argv[i + 5]);
            i += 5;
        }
        else if (string(argv[i]) == "--dist-left") {
            dist_left = Mat::zeros(1, 5, CV_64F);
            dist_left(0, 0) = atof(argv[i + 1]);
            dist_left(0, 1) = atof(argv[i + 2]);
            dist_left(0, 2) = atof(argv[i + 3]);
            dist_left(0, 3) = atof(argv[i + 4]);
            dist_left(0, 4) = atof(argv[i + 5]);
            i += 5;
        }
        else if (string(argv[i]) == "--dist-right") {
            dist_right = Mat::zeros(1, 5, CV_64F);
            dist_right(0, 0) = atof(argv[i + 1]);
            dist_right(0, 1) = atof(argv[i + 2]);
            dist_right(0, 2) = atof(argv[i + 3]);
            dist_right(0, 3) = atof(argv[i + 4]);
            dist_right(0, 4) = atof(argv[i + 5]);
            i += 5;
        }
        else {
            img_names.push_back(argv[i]);
        }
        CV_Assert(img_names.size() % 2 == 0);
        CV_Assert(!K_left.empty());
        CV_Assert(!K_right.empty());
        CV_Assert(!dist_left.empty());
        CV_Assert(!dist_right.empty());
    }
}
