#include <iostream>
#include <string>
#include <stdexcept>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    try {
        if (argc < 2)
            throw runtime_error("Specify frames stream");

        VideoCapture vc(argv[1]);
        if (!vc.isOpened())
            throw runtime_error("Can't open " + string(argv[1]));

        Mat frame;
        vc >> frame;
        for (; !frame.empty(); vc >> frame) {
            imshow("f", frame);
            waitKey();
        }
    }
    catch (const exception &e) {
        cout << "Error: " << e.what() << endl;
    }
    return 0;
}
