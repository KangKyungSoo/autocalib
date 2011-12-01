#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void ConvertRgbToBgr(const Mat &src, Mat &dst);

int main(int argc, char **argv) {
    if (argc < 2) {
        cout << "Usage: autocalib_takeshots <camera_id>\n";
        return 1;
    }
    try {
        int camera_id = atoi(argv[1]);
        VideoCapture vc(camera_id);
        Mat left, right;
        bool ok = vc.grab();
        int shot_idx = 0;

        while (ok) {
            vc.retrieve(left, 0);
            vc.retrieve(right, 1);
            ConvertRgbToBgr(left, left);
            ConvertRgbToBgr(right, right);
            imshow("left", left);
            imshow("right", right);

            int key = waitKey(3);
            if (key == 't' || key == 'T') {
                stringstream ss;
                ss << "shot_" << shot_idx << "_left.jpg";
                imwrite(ss.str(), left);
                cout << "took shot " << ss.str() << endl;
                ss.str("");
                ss << "shot_" << shot_idx << "_right.jpg";
                imwrite(ss.str(), right);
                cout << "took shot " << ss.str() << endl;
                shot_idx++;
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
