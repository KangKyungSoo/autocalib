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
