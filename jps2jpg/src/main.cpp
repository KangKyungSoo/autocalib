#include <iostream>
#include <string>
#include <sstream>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    try {
        for (int i = 1; i < argc; ++i) {
            string name = argv[i];
            Mat stereo_frame = imread(name);
            if (stereo_frame.empty())
                throw runtime_error("Can't open " + name);
            name = name.substr(0, name.find_last_of('.'));

            Mat left = stereo_frame.colRange(0, stereo_frame.cols / 2);
            stringstream left_name;
            left_name << "mono_" << name << "_l.jpg";
            imwrite(left_name.str(), left);

            Mat right = stereo_frame.colRange(stereo_frame.cols / 2, stereo_frame.cols);
            stringstream right_name;
            right_name << "mono_" << name << "_r.jpg";
            imwrite(right_name.str(), right);
        }
    }
    catch (const exception &e) {
        cout << "Error: " << e.what() << endl;
    }
    return 0;
}