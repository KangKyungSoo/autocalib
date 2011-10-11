#include <iostream>
#include <vector>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <core/include/core.h>

using namespace std;
using namespace cv;
using namespace autocalib;

vector<Mat> imgs;

void ParseArgs(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        Mat img = imread(argv[++i]);
        if (img.empty())
            throw runtime_error(string("Can't open image: ") + argv[i]);
    }
}


int main(int argc, char **argv) {
    try {
        ParseArgs(argc, argv);
    }
    catch (const exception &e) {
        cout << "Error: " << e.what() << endl;
    }
    return 0;
}
