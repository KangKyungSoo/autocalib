#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

vector<string> all_image_names;
string app_name;
string app_args;
enum AppType {
    APP_TYPE_UNKNOWN,
    APP_TYPE_OPENCV,
    APP_TYPE_AUTOCALIB
};
AppType app_type = APP_TYPE_UNKNOWN;


void ParseArgs(int argc, char **argv);
void RunOpencvApp(const vector<string> &image_names);
void RunAutocalibApp(const vector<string> &image_names);


int main(int argc, char **argv) {
    try {
        ParseArgs(argc, argv);

        CV_Assert(!all_image_names.empty());
        CV_Assert(all_image_names.size() % 2 == 0);
        CV_Assert(!app_name.empty());
        CV_Assert(app_type != APP_TYPE_UNKNOWN);

        for (size_t i = 0; i < all_image_names.size(); i += 2) {
            vector<string> image_names;
            for (size_t j = 0; j < all_image_names.size(); j += 2) {
                if (j != i) {
                    image_names.push_back(all_image_names[j]);
                    image_names.push_back(all_image_names[j + 1]);
                }
            }
            if (app_type == APP_TYPE_OPENCV) {
                RunOpencvApp(image_names);
            }
            else if (app_type == APP_TYPE_AUTOCALIB) {
                RunAutocalibApp(image_names);
            }
        }
    }
    catch (const exception &e) {
        cout << "Error: " << e.what() << endl;
    }
    return 0;
}


void ParseArgs(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--app-name")
            app_name = argv[++i];
        else if (string(argv[i]) == "--app-args")
            app_args = argv[++i];
        else if (string(argv[i]) == "--app-type") {
            if (string(argv[i + 1]) == "opencv")
                app_type = APP_TYPE_OPENCV;
            else if (string(argv[i + 1]) == "autocalib")
                app_type = APP_TYPE_AUTOCALIB;
            else
                throw runtime_error("unknown calibration application type");
            ++i;
        }
        else
            all_image_names.push_back(argv[i]);
    }
}


void RunOpencvApp(const vector<string> &image_names) {
    ofstream image_list_file("image_list.xml");
    image_list_file << "<?xml version=\"1.0\"?><opencv_storage><images>\n";
    for (size_t i = 0; i < image_names.size(); ++i)
        image_list_file << image_names[i] << endl;
    image_list_file << "</images></opencv_storage>";
    image_list_file.close();

    stringstream cmd;
    cmd << app_name << " image_list.xml " << app_args;
    cout << "COMMAND: " << cmd.str() << endl;
    system(cmd.str().c_str());

    FileStorage extrinsics_file("extrinsics.yml", FileStorage::READ);
    CV_Assert(extrinsics_file.isOpened());

    Mat_<double> T;
    extrinsics_file["T"] >> T;
    T /= T(0, 0);

    Mat_<double> R;
    extrinsics_file["R"] >> R;
    Mat_<double> rvec;
    Rodrigues(R, rvec);

    FileStorage intrinsics_file("intrinsics.yml", FileStorage::READ);
    CV_Assert(intrinsics_file.isOpened());

    Mat_<double> K;
    intrinsics_file["M1"] >> K;

    ofstream log_file("opencv_log.csv", ios_base::app);
    log_file << T(0, 0) << ";" << T(0, 1) << ";" << T(0, 2) << ";"
             << rvec(0, 0) << ";" << rvec(1, 0) << ";" << rvec(2, 0) << ";"
             << K(0, 0) << ";" << K(1, 1) << ";" << K(0, 2) << ";" << K(1, 2) << ";" << K(0, 1) << ";";

    for (size_t i = 0; i < image_names.size(); ++i)
        log_file << image_names[i] << " ";
    log_file << endl;
}


void RunAutocalibApp(const vector<string> &image_names) {
    stringstream cmd;
    cmd << app_name << " ";
    for (size_t i = 0; i < image_names.size(); ++i)
        cmd << image_names[i] << " ";
    cmd << app_args;
    cout << "COMMAND: " << cmd.str() << endl;
    system(cmd.str().c_str());

    FileStorage params_file("autocalib_camera_params.yml", FileStorage::READ);
    CV_Assert(params_file.isOpened());

    Mat_<double> rvec;
    params_file["rvec_est"] >> rvec;

    Mat_<double> T;
    params_file["T_est"] >> T;
    T /= T(0, 0);

    Mat_<double> K;
    params_file["K_est"] >> K;

    ofstream log_file("autocalib_log.csv", ios_base::app);
    log_file << T(0, 0) << ";" << T(0, 1) << ";" << T(0, 2) << ";"
             << rvec(0, 0) << ";" << rvec(1, 0) << ";" << rvec(2, 0) << ";"
             << K(0, 0) << ";" << K(1, 1) << ";" << K(0, 2) << ";" << K(1, 2) << ";" << K(0, 1) << ";";

    for (size_t i = 0; i < image_names.size(); ++i)
        log_file << image_names[i] << " ";
    log_file << endl;
}
