#include "precomp.h"
#include <core/include/core.h>

using namespace std;
using namespace cv;
using namespace autocalib;


TEST(Anitdiag, SquareIsUnit) {
    Mat A = Antidiag(3, 3, CV_64F);

    ASSERT_TRUE(A.size() == Size(3, 3));
    ASSERT_EQ(CV_64F, A.type());
    ASSERT_LE(norm(Mat::eye(3, 3, CV_64F), A * A), 1e-6);
}


TEST(DecomposeCholesky, CanDecomposeSmallMatrix) {
    Mat_<double> L(3, 3);
    L(0, 0) = 1;
    L(1, 0) = 2; L(1, 1) = 3;
    L(2, 0) = 4; L(2, 1) = 5; L(2, 2) = 6;

    Mat dst = DecomposeCholesky(L * L.t());

    ASSERT_TRUE(!dst.empty());
    ASSERT_LT(norm(dst, L, NORM_INF), 1e-6);
}


TEST(DecomposeCholesky, CanNotDecomposeNegativeDefiniteMatrix) {
    Mat_<double> L(3, 3);
    L(0, 0) = 1;
    L(1, 0) = 2; L(1, 1) = 3;
    L(2, 0) = 4; L(2, 1) = 5; L(2, 2) = 6;

    ASSERT_TRUE(DecomposeCholesky(-L * L.t()).empty());
}


TEST(DecomposeUUt, CanDecomposeSmallMatrix) {
    Mat_<double> U(3, 3);
    U(0, 0) = 1; U(0, 1) = 2; U(0, 2) = 3;
    U(1, 1) = 4; U(1, 2) = 5;
    U(2, 2) = 6;

    Mat dst = DecomposeUUt(U * U.t());

    ASSERT_TRUE(!dst.empty());
    ASSERT_LT(norm(dst, U, NORM_INF), 1e-3);
}


TEST(TruncEigenvals, CanDoNotTruncation) {
    Mat_<double> L(3, 3);
    L(0, 0) = 1;
    L(1, 0) = 2; L(1, 1) = 3;
    L(2, 0) = 4; L(2, 1) = 5; L(2, 2) = 6;

    Mat src = L * L.t();
    Mat dst = TruncEigenvals(src, Interval::All());

    ASSERT_TRUE(src.size() == dst.size());
    ASSERT_TRUE(src.type() == CV_64F);
    ASSERT_LT(norm(dst, src, NORM_INF), 1e-6);
}


TEST(TruncEigenvals, CanTruncNegativeEval) {
    Mat_<double> L(3, 3);
    L(0, 0) = 1;
    L(1, 0) = 2; L(1, 1) = 3;
    L(2, 0) = 4; L(2, 1) = 5; L(2, 2) = 6;

    Mat_<double> eigenvals, eigenvecs;

    eigen(L * L.t(), eigenvals, eigenvecs);
    eigenvals(2, 0) = -1;
    Mat src = eigenvecs.t() * Mat::diag(eigenvals) * eigenvecs;

    Mat dst = TruncEigenvals(src, Interval::Left(1e-6));
    eigen(dst, eigenvals, eigenvecs);

    ASSERT_TRUE(src.size() == dst.size());
    ASSERT_TRUE(src.type() == CV_64F);
    ASSERT_GT(eigenvals(0, 0), 0);
    ASSERT_GT(eigenvals(1, 0), 0);
    ASSERT_GT(eigenvals(2, 0), 0);
}


TEST(TruncEigenvals, CanTruncPositiveEval) {
    Mat_<double> L(3, 3);
    L(0, 0) = 1;
    L(1, 0) = 2; L(1, 1) = 3;
    L(2, 0) = 4; L(2, 1) = 5; L(2, 2) = 6;

    Mat_<double> eigenvals, eigenvecs;

    eigen(-L * L.t(), eigenvals, eigenvecs);
    eigenvals(2, 0) = 1;
    Mat src = eigenvecs.t() * Mat::diag(eigenvals) * eigenvecs;

    Mat dst = TruncEigenvals(src, Interval::Right(-1e-6));
    eigen(dst, eigenvals, eigenvecs);

    ASSERT_TRUE(src.size() == dst.size());
    ASSERT_TRUE(src.type() == CV_64F);
    ASSERT_LT(eigenvals(0, 0), 0);
    ASSERT_LT(eigenvals(1, 0), 0);
    ASSERT_LT(eigenvals(2, 0), 0);
}


TEST(TruncEigenvals, CanTruncUsingInterval) {
    Mat_<double> L(3, 3);
    L(0, 0) = 1;
    L(1, 0) = 2; L(1, 1) = 3;
    L(2, 0) = 4; L(2, 1) = 5; L(2, 2) = 6;

    Mat_<double> eigenvals, eigenvecs;

    eigen(L * L.t(), eigenvals, eigenvecs);
    eigenvals(0, 0) = -1;
    eigenvals(1, 0) = 0;
    eigenvals(2, 0) = 1;
    Mat src = eigenvecs.t() * Mat::diag(eigenvals) * eigenvecs;

    Mat dst = TruncEigenvals(src, Interval(-0.5, 0.5));
    eigen(dst, eigenvals, eigenvecs);

    ASSERT_TRUE(src.size() == dst.size());
    ASSERT_TRUE(src.type() == CV_64F);
    ASSERT_NEAR(0.5, eigenvals(0, 0), 1e-6);
    ASSERT_NEAR(0, eigenvals(1, 0), 1e-6);
    ASSERT_NEAR(-0.5, eigenvals(2, 0), 1e-6);
}


TEST(Quaternion, EyeToQuaternion) {
    Mat R = Mat::eye(3, 3, CV_64F);
    Quaternion q = Quaternion::FromRotationMat(R);

    ASSERT_DOUBLE_EQ(1, q[0]);
    ASSERT_DOUBLE_EQ(0, q[1]);
    ASSERT_DOUBLE_EQ(0, q[2]);
    ASSERT_DOUBLE_EQ(0, q[3]);
}


TEST(Quaternion, QuaternionToEye) {
    Quaternion q(1, 0, 0, 0);
    Mat R = q.RotationMat();

    ASSERT_LE(norm(Mat::eye(3, 3, CV_64F), R, NORM_INF), 1e-6);
}


TEST(Quaternion, QuaternionToMat) {
    Quaternion q;
    Mat R;
    RNG rng(0);

    for (int i = 0; i < 1000; ++i) {
        q[0] = (double)rng; q[1] = (double)rng; q[2] = (double)rng; q[3] = (double)rng;
        q /= q.Norm();
        R = q.RotationMat();

        ASSERT_NEAR(1, determinant(R), 1e-6);
        ASSERT_LE(norm(R * R.t(), Mat::eye(3, 3, CV_64F), NORM_INF), 1e-6);
    }
}


TEST(Quaternion, QuaternionToMatToQuaternion) {
    Quaternion q, q_;
    Mat R;
    RNG rng(0);

    for (int i = 0; i < 1000; ++i) {
        q[0] = (double)rng; q[1] = (double)rng; q[2] = (double)rng; q[3] = (double)rng;
        q /= q.Norm();
        R = q.RotationMat();
        q_ = Quaternion::FromRotationMat(R);

        ASSERT_NEAR(abs(q[0]), abs(q_[0]), 1e-6);
        double sign = q[0] * q_[0] > 0 ? 1 : -1;
        ASSERT_NEAR(q[1], sign * q_[1], 1e-6);
        ASSERT_NEAR(q[2], sign * q_[2], 1e-6);
        ASSERT_NEAR(q[3], sign * q_[3], 1e-6);
    }
}


TEST(Quaternion, Multiplication) {
    Quaternion q1, q2, q3;
    Mat R1, R2, R3, R3_;
    RNG rng(0);

    for (int i = 0; i < 1000; ++i) {
        q1[0] = (double)rng; q1[1] = (double)rng; q1[2] = (double)rng; q1[3] = (double)rng;
        q2[0] = (double)rng; q2[1] = (double)rng; q2[2] = (double)rng; q2[3] = (double)rng;
        q1 /= q1.Norm();
        q2 /= q2.Norm();
        q3 = q1 * q2;
        R1 = q1.RotationMat();
        R2 = q2.RotationMat();
        R3 = R1 * R2;
        R3_ = q3.RotationMat();

        ASSERT_NEAR(0, norm(R3, R3_, NORM_INF), 1e-6);
    }
}


TEST(Quaternion, Reciprocal) {
    Quaternion q;
    RNG rng(0);

    for (int i = 0; i < 1000; ++i) {
        q[0] = (double)rng; q[1] = (double)rng; q[2] = (double)rng; q[3] = (double)rng;

        ASSERT_NEAR(0, (q * q.Reciprocal() - 1).Norm(), 1e-6);
        ASSERT_NEAR(0, (q.Reciprocal() * q - 1).Norm(), 1e-6);
    }
}
