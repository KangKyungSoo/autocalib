#include "precomp.h"
#include <core/include/core.h>

using namespace std;
using namespace cv;
using namespace autocalib;


TEST(Anitdiag, SquareIsUnit) {
    Mat A = Antidiag(3, 3, CV_64F);

    ASSERT_TRUE(A.size() == Size(3, 3));
    ASSERT_EQ(CV_64F, A.type());
    ASSERT_LE(norm(Mat::eye(3, 3, CV_64F), A * A), 1e-3);
}


TEST(DecomposeCholesky, CanDecomposeSmallMatrix) {
    Mat_<double> L(3, 3);
    L(0, 0) = 1;
    L(1, 0) = 2; L(1, 1) = 3;
    L(2, 0) = 4; L(2, 1) = 5; L(2, 2) = 6;

    Mat dst = DecomposeCholesky(L * L.t());

    ASSERT_TRUE(!dst.empty());
    ASSERT_LT(norm(dst, L, NORM_INF), 1e-3);
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
    ASSERT_LT(norm(dst, src, NORM_INF), 1e-3);
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

    Mat dst = TruncEigenvals(src, Interval::Left(1e-3));
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

    Mat dst = TruncEigenvals(src, Interval::Right(-1e-3));
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
