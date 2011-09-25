#include "precomp.h"
#include <include/core.h>

using namespace std;
using namespace cv;

namespace autocalib {

Mat CalibRotationalCameraLinear(InputArrayOfArrays Hs) {
    vector<Mat> Hs_;
    Hs.getMatVector(Hs_);
    int num_Hs = (int)Hs_.size();
    if (num_Hs < 1)
        throw runtime_error("Need at least one homography");

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
                A(eq_idx, 1) = H(r1, 0) * H(r2, 1) + H(r1, 1) * H(r2, 0);
                A(eq_idx, 2) = H(r1, 0) * H(r2, 2) + H(r1, 2) * H(r2, 0);
                A(eq_idx, 3) = H(r1, 1) * H(r2, 1);
                A(eq_idx, 4) = H(r1, 1) * H(r2, 2) + H(r1, 2) * H(r2, 1);
                if (r1 != 2 && r1 != 2) {
                    A(eq_idx, lut[r1][r2]) -= 1;
                    b(eq_idx, 0) = -H(r1, 2) * H(r2, 2);
                }
                else
                    b(eq_idx, 0) = 1 - H(r1, 2) * H(r2, 2);
                eq_idx++;
            }
        }
    }

    Mat_<double> x;
    solve(A, b, x, DECOMP_SVD);
    Mat_<double> KK = Mat::eye(3, 3, CV_64F);
    KK(0, 0) = x(0, 0);
    KK(0, 1) = KK(1, 0) = x(1, 0);
    KK(0, 2) = KK(2, 0) = x(2, 0);
    KK(1, 1) = x(3, 0);
    KK(1, 2) = KK(2, 1) = x(4, 0);

    // Do U * U.t() decomposition
    Mat adiag = Antidiag(3, 3, CV_64F);
    Mat K_flipped = DecomposeCholesky(adiag * KK * adiag);
    if (K_flipped.empty())
        throw runtime_error("K * K.t() isn't positive definite");
    return adiag * K_flipped * adiag;
}


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


Mat DecomposeCholesky(InputArray src) {
    Mat src_ = src.getMat();
    CV_Assert(src_.rows == src_.cols && src_.type() == CV_64F);

    Mat L;
    src_.copyTo(L);

    if (!Cholesky(L.ptr<double>(), L.step, L.cols, 0, 0, 0))
        return Mat();

    for (int i = 0; i < L.cols; ++i)
        for (int j = i + 1; j < L.rows; ++j)
            L.at<double>(i, j) = 0;

    for (int i = 0; i < L.cols; ++i)
        L.at<double>(i, i) = 1. / L.at<double>(i, i);

    return L;
}


void ExtractMatchedKeypoints(const detail::ImageFeatures &f1, const detail::ImageFeatures &f2,
                             const vector<DMatch> &matches, OutputArray kps1, OutputArray kps2)
{
    Mat &kps1_ = kps1.getMatRef();
    Mat &kps2_ = kps2.getMatRef();

    kps1_.create(1, (int)matches.size(), CV_32FC2);
    kps2_.create(1, (int)matches.size(), CV_32FC2);

    for (size_t i = 0; i < matches.size(); ++i) {
        kps1_.at<Point2f>(0, i) = f1.keypoints[matches[i].queryIdx].pt;
        kps2_.at<Point2f>(0, i) = f2.keypoints[matches[i].trainIdx].pt;
    }
}

} // namespace autocalib
