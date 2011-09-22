#include "precomp.h"
#include <include/core.h>

using namespace std;
using namespace cv;

namespace autocalib {

Mat Antidiag(int rows, int cols, int type) {
    Mat dst = Mat::zeros(rows, cols, type);
    int len = min(rows, cols);

    switch (type) {
    case CV_8U:
        for (int i = 0; i < len; ++i)
            dst.at<uchar>(i, cols - i - 1) = 1;
        break;
    case CV_16S:
        for (int i = 0; i < len; ++i)
            dst.at<short>(i, cols - i - 1) = 1;
        break;
    case CV_32S:
        for (int i = 0; i < len; ++i)
            dst.at<int>(i, cols - i - 1) = 1;
        break;
    case CV_32F:
        for (int i = 0; i < len; ++i)
            dst.at<float>(i, cols - i - 1) = 1.f;
        break;
    case CV_64F:
        for (int i = 0; i < len; ++i)
            dst.at<double>(i, cols - i - 1) = 1;
        break;
    }

    return dst;
}


bool DecomposeCholesky(InputArray src, OutputArray L) {
    Mat src_ = src.getMat();
    CV_Assert(src_.rows == src_.cols && src_.type() == CV_64F);

    Mat &L_ = L.getMatRef();
    src_.copyTo(L_);

    if (!Cholesky(L_.ptr<double>(), L_.step, L_.cols, 0, 0, 0))
        return false;

    for (int i = 0; i < L_.cols; ++i)
        for (int j = i + 1; j < L_.rows; ++j)
            L_.at<double>(i, j) = 0;

    for (int i = 0; i < L_.cols; ++i)
        L_.at<double>(i, i) = 1. / L_.at<double>(i, i);

    return true;
}


Mat CalibRotationalCameraLinear(InputArrayOfArrays Hs) {
    vector<Mat> Hs_;
    Hs.getMatVector(Hs_);
    int num_Hs = static_cast<int>(Hs_.size());

    // Ensure all homographies has unit determinant
    vector<Mat> Hs_normed(num_Hs);
    for (int i = 0; i < num_Hs; ++i) {
        CV_Assert(Hs_[i].size() == Size(3, 3) && Hs_[i].type() == CV_64F);
        Hs_normed[i] = Hs_[i] / pow(determinant(Hs_[i]), 1. / 3.);
    }

    Mat_<double> A(6 * num_Hs, 5);
    Mat_<double> b(6 * num_Hs, 1);
    b.setTo(0);

    static const int lut[][3] = {{0, 1, 2}, {-1, 3, 4}, {-1, -1, -1}};

    int eq_idx = 0;
    for (int H_idx = 0; H_idx < num_Hs; ++H_idx) {
        Mat_<double> H = Hs_normed[H_idx];
        for (int r1 = 0; r1 < 3; ++r1) {
            for (int r2 = r1; r2 < 3; ++r2) {
                A(eq_idx, 0) = H(r1, 0) * H(r2, 0);
                A(eq_idx, 1) = H(r1, 0) * H(r2, 1) * 2;
                A(eq_idx, 2) = H(r1, 0) * H(r2, 2) * 2;
                A(eq_idx, 3) = H(r1, 1) * H(r2, 1);
                A(eq_idx, 4) = H(r1, 1) * H(r2, 2) * 2;
                A(eq_idx, 5) = H(r1, 2) * H(r2, 2);
                if (r1 != 2 && r1 != 2)
                    A(eq_idx, lut[r1][r2]) -= 1;
                else
                    b(eq_idx, 5) = 1;
                eq_idx++;
            }
        }
    }

    Mat_<double> x;
    solve(A, b, x, DECOMP_SVD);
    Mat_<double> KK = Mat::eye(3, 3, CV_64F);
    KK(0, 0) = x(0, 0); KK(0, 1) = x(1, 0); KK(0, 2) = x(2, 0);
    KK(1, 1) = x(3, 0); KK(1, 2) = x(4, 0);

    // Do U * U.t() decomposition
    Mat K_flipped;
    Mat adiag = Antidiag(3, 3, CV_64F);
    DecomposeCholesky(adiag * KK * adiag, K_flipped);
    return adiag * K_flipped * adiag;
}

} // namespace autocalib
