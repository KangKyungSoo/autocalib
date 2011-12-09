#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void ConvertRgbToBgr(const Mat &src, Mat &dst);
void ParseCmdArgs(int argc, char **argv);

int cam_id = 0;
Mat_<double> K_left, K_right;
Mat_<double> dist_left, dist_right;
bool do_undist = false;

int main(int argc, char **argv) {
    try {
        ParseCmdArgs(argc, argv);

        VideoCapture vc(cam_id);
        Mat left, right;
        Mat left_undist, right_undist;
        bool ok = vc.grab();
        int shot_idx = 0;

        while (ok) {
            vc.retrieve(left, 0);
            vc.retrieve(right, 1);
            ConvertRgbToBgr(left, left);
            ConvertRgbToBgr(right, right);

            if (do_undist) {
                undistort(left, left_undist, K_left, dist_left);
                undistort(right, right_undist, K_right, dist_right);
            }

            if (do_undist) {
                imshow("left", left_undist);
                imshow("right", right_undist);
            }
            else {
                imshow("left", left);
                imshow("right", right);
            }

            int key = waitKey(3);
            if (key == 't' || key == 'T') {
                stringstream ss;
                ss << "shot_" << shot_idx << "_left.jpg";
                imwrite(ss.str(), left);
                cout << "took shot " << ss.str() << endl;
                if (do_undist) {
                    ss.str("");
                    ss << "undist_shot_" << shot_idx << "_left.jpg";
                    imwrite(ss.str(), left_undist);
                    cout << "took shot " << ss.str() << endl;
                }
                ss.str("");
                ss << "shot_" << shot_idx << "_right.jpg";
                imwrite(ss.str(), right);
                cout << "took shot " << ss.str() << endl;
                if (do_undist) {
                    ss.str("");
                    ss << "undist_shot_" << shot_idx << "_right.jpg";
                    imwrite(ss.str(), right_undist);
                    cout << "took shot " << ss.str() << endl;
                }
                shot_idx++;
            }
            else if (key == 'd' || key == 'D') {
                do_undist = !do_undist;
                cout << "do undistort = " << do_undist << endl;
            }
            else if (key == 27) {
                break;
            }

            ok = vc.grab();
        }
    }
    catch (const exception &e) {
        cout << "Error: " << e.what() << endl;
    }
    return 0;
}


void ConvertRgbToBgr(const Mat &src, Mat &dst) {
    vector<Mat> planes(3);
    split(src, planes);
    vector<Mat> planes_reordered(3);
    planes_reordered[0] = planes[2];
    planes_reordered[1] = planes[1];
    planes_reordered[2] = planes[0];
    merge(planes_reordered, dst);
}


void ParseCmdArgs(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--cam-id")
            cam_id = atoi(argv[++i]);
        else if (string(argv[i]) == "--K-left") {
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
        else
            throw runtime_error("Can't parse the following flag: " + string(argv[i]));
    }
}
