#ifndef AUTOCALIB_CORE_H_
#define AUTOCALIB_CORE_H_

#include <vector>
#include <map>
#include <string>
#include <utility>
#include <limits>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <config.h>

namespace autocalib {

//============================================================================
// Cameras

/** General projective camera interface. */
class IProjectiveCamera {
public:
    virtual ~IProjectiveCamera() {}

    /** \return 4x3 projective camera matrix */
    virtual const cv::Mat P() const = 0;
};


/** Describes a projective camera. */
class ProjectiveCamera : public IProjectiveCamera {
public:

    /** Construct a projective camera from 3x4 matrix. */
    ProjectiveCamera(const cv::Mat &P) {
        CV_Assert(P.size() == cv::Size(4, 3) && P.type() == CV_64F);
        P_ = P.clone();
    }

    virtual const cv::Mat P() const { return P_; }

private:
    cv::Mat P_;
};


/** Describes a rigid camera. */
class RigidCamera : public IProjectiveCamera {
public:

    /** Creates a camera from intrinsics and mapping from local to world coordinates.
      *
      * \param K Intrinsics matrix
      * \param R Rotation matrix
      * \param center Camera center
      * \return Camera object
      */
    static RigidCamera LocalToWorld(const cv::Mat &K, const cv::Mat &R, const cv::Mat &center) {
        cv::Mat R_inv = R.inv();
        return RigidCamera(K, R_inv, -R_inv * center);
    }

    /** Default constructor. Creates an eye camara. */
    RigidCamera() {
        K_ = cv::Mat::eye(3, 3, CV_64F);
        R_ = cv::Mat::eye(3, 3, CV_64F);
        T_ = cv::Mat::zeros(3, 1, CV_64F);
    }

    /** Constructs a camera from intrinsics and mapping from world to local coordinates
      * (extrinsics).
      *
      * \param K Intrinsics matrix
      * \param R Rotation matrix
      * \param T Translation vector
      */
    RigidCamera(const cv::Mat &K, const cv::Mat &R, const cv::Mat &T) {
        CV_Assert(K.size() == cv::Size(3, 3) && K.type() == CV_64F);
        CV_Assert(R.size() == cv::Size(3, 3) && R.type() == CV_64F);
        CV_Assert(T.size() == cv::Size(1, 3) && T.type() == CV_64F);
        K_ = K.clone();
        R_ = R.clone();
        T_ = T.clone();
    }

    virtual const cv::Mat P() const {
        cv::Mat result(3, 4, CV_64F);
        cv::Mat tmp = result(cv::Rect(0, 0, 3, 3));
        cv::Mat(K_ * R_).copyTo(tmp);
        tmp = result(cv::Rect(3, 0, 1, 3));
        cv::Mat(K_ * T_).copyTo(tmp);
        return result;
    }

    const cv::Mat& K() const { return K_; }
    const cv::Mat& R() const { return R_; }
    const cv::Mat& T() const { return T_; }

private:    

    cv::Mat K_, R_, T_;
};


//============================================================================
// Optimization

/** Minimization method options. */
class MinimizeOpts {
public:

    /** Verbosity level. */
    enum Verbose {
        VERBOSE_NO = 0,
        VERBOSE_SUMMARY = 1,
        VERBOSE_ITER = 2
    };

    /** \param crit Termination criteria
      * \param verbose Verbosity level
      * \see Verbose
      */
    MinimizeOpts(cv::TermCriteria crit = crit_default(), int verbose = VERBOSE_NO) {
        Init(crit, verbose);
    }

    /** Constructs options using the default termination criteria.
      *
      * \param verbose Verbosity level
      * \see Verbose
      */
    MinimizeOpts(int verbose) { Init(crit_default(), verbose); }

    /** \return Default termination criteria. */
    static cv::TermCriteria crit_default() {
        return cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                                1000, std::numeric_limits<double>::epsilon());
    }

    const int verbose() const { return verbose_; }
    const cv::TermCriteria& crit() const { return crit_; }

private:
    void Init(cv::TermCriteria term_crit, int verbose) {
        crit_ = term_crit;
        verbose_ = verbose;
    }

    cv::TermCriteria crit_;
    int verbose_;
};


/** Minimizes a function using the Levenberg-Marquardt algorithm.
  *
  * \param func Function to be minimized
  * \param arg Function arguments
  * \param opts Minimization method options
  * \return L2 norm of optimal value
  * \see MinimizeOpts
  */
template <typename Func>
double MinimizeLevMarq(Func func, cv::InputOutputArray arg, MinimizeOpts opts = MinimizeOpts());


//============================================================================
// Autocalibration

typedef std::vector<cv::detail::ImageFeatures> FeaturesCollection;
typedef std::map<std::pair<int, int>, std::vector<cv::DMatch> > MatchesCollection;


/** Describes an interval. */
struct Interval {

    /** Interval kind. */
    enum Kind { ALL, LEFT, RIGHT, LEFT_RIGHT };

    /** Creates (left, right) interval.
      *
      * \return Interval
      */
    Interval(double left, double right) {
        kind_ = LEFT_RIGHT;
        left_ = left;
        right_ = right;
    }

    /** Creates (-inf, +inf) interval.
      *
      * \return Interval
      */
    static Interval All() {
        Interval res;
        res.kind_ = ALL;
        return res;
    }

    /** Creates (left, +inf) interval.
      *
      * \return Interval
      */
    static Interval Left(double left) {
        Interval res;
        res.kind_ = LEFT;
        res.left_ = left;
        return res;
    }

    /** Creates (-inf, right) interval.
      *
      * \return Interval
      */
    static Interval Right(double right) {
        Interval res;
        res.kind_ = RIGHT;
        res.right_ = right;
        return res;
    }

    Kind kind() const { return kind_; }
    double left() const { return left_; }
    double right() const { return right_; }

private:
    Interval() {}

    Kind kind_;
    double left_;
    double right_;
};


/** Calculates rotational camera intrinsics using a linear algorithm.
  *
  * See details in Hartey R., Zisserman A., "Multiple View Geometry", 2nd ed., p. 482.
  *
  * \param Hs Projective plane homographies
  * \param evals_interval Interval used for DIAC eigenvalues truncation (true eigenvalues are [fx^2, fy^2, 1])
  * \return Camera intrinsics
  */
cv::Mat CalibRotationalCameraLinear(cv::InputArrayOfArrays Hs,
                                    Interval evals_interval = Interval::All());


/** Calculates rotational camera intrinsics using a linear algorithm with the zero skew assumption.
  *
  * See details in Hartey R., Zisserman A., "Multiple View Geometry", 2nd ed., p. 482.
  *
  * \param Hs Projective plane homographies
  * \param evals_interval Interval used for IAC eigenvalues truncation (true eigenvalues are [(1/fx)^2, (1/fy)^2, 1])
  * \return Camera intrinsics, where skew is zero
  */
cv::Mat CalibRotationalCameraLinearNoSkew(cv::InputArrayOfArrays Hs,
                                          Interval evals_interval = Interval::All());


/** Describes a quaternion of the following form: a + b*i + c*j + d*k. */
class Quaternion {
public:
    Quaternion() {}
    Quaternion(const Quaternion &q) { a_ = q.a_; b_ = q.b_; c_ = q.c_; d_ = q.d_; }

    /** Creates a quaternion of the following form: a + b*i + c*j + d*k. */
    Quaternion(double a, double b, double c, double d) { a_ = a; b_ = b; c_ = c; d_ = d; }

    double a() const { return a_; }
    double& a() { return a_; }

    double b() const { return b_; }
    double& b() { return b_; }

    double c() const { return c_; }
    double& c() { return c_; }

    double d() const { return d_; }
    double& d() { return d_; }

    double operator [](int index) const { return vals_[index]; }
    double& operator [](int index) { return vals_[index]; }

    Quaternion Conjuagate() const { return Quaternion(a_, -b_, -c_, -d_); }

    Quaternion Reciprocal() const { return Conjuagate() /= SquaredNorm(); }

    double SquaredNorm() const { return a_ * a_ + b_ * b_ + c_ * c_ + d_ * d_; }

    double Norm() const { return std::sqrt(SquaredNorm()); }

    const Quaternion& operator +=(const Quaternion &q) {
        a_ += q.a_; b_ += q.b_; c_ += q.c_; d_ += q.d_;
        return *this;
    }

    const Quaternion& operator -=(const Quaternion &q) {
        a_ -= q.a_; b_ -= q.b_; c_ -= q.c_; d_ -= q.d_;
        return *this;
    }

    const Quaternion& operator *=(double a) {
        a_ *= a; b_ *= a; c_ *= a; d_ *= a;
        return *this;
    }

    const Quaternion& operator *=(const Quaternion &q) {
        return *this = *this * q;
    }

    const Quaternion& operator /=(double a) {
        return *this *= 1 / a;
    }

    Quaternion operator +(const Quaternion &q) const { return Quaternion(*this) += q; }

    Quaternion operator -(const Quaternion &q) const { return Quaternion(*this) -= q; }

    Quaternion operator *(double a) const { return Quaternion(*this) *= a; }

    Quaternion operator *(const Quaternion &q) const {
        return Quaternion(a_ * q.a_ - b_ * q.b_ - c_ * q.c_ - d_ * q.d_,
                          a_ * q.b_ + b_ * q.a_ + c_ * q.d_ - d_ * q.c_,
                          a_ * q.c_ - b_ * q.d_ + c_ * q.a_ + d_ * q.b_,
                          a_ * q.d_ + b_ * q.c_ - c_ * q.b_ + d_ * q.a_);
    }

    Quaternion operator /(double a) const { return Quaternion(*this) /= a; }

private:
    union {
        double a_, b_, c_, d_;
        double vals_[4];
    };
};


/** Rigid camera refinement method flags. */
enum RefineFlag {
    REFINE_FLAG_FX = 1,
    REFINE_FLAG_FY = 2,
    REFINE_FLAG_PPX = 4,
    REFINE_FLAG_PPY = 8,
    REFINE_FLAG_SKEW = 16,
    REFINE_FLAG_ALL = 31
};


/** Refines rigid camera parameters by minimizing overall reprojection error.
  *
  * \param K Camera intrinsics
  * \param Rs Camera positions vector
  * \param features Features collection
  * \param matches Matches collection
  * \param params_to_refine Flags indicating parameters which should be refined
  * \see RefineFlag
  */
void RefineRigidCamera(cv::InputOutputArray K, cv::InputOutputArrayOfArrays Rs,
                       const FeaturesCollection &features, const MatchesCollection &matches,
                       int params_to_refine = REFINE_FLAG_ALL);


//============================================================================
// Other

/** Constructs anti-diagonal matrix of ones.
  *
  * \param rows Number of rows
  * \param cols Number of cols
  * \param type Matrix type
  * \return Anti-diagonal matrix
  */
cv::Mat Antidiag(int rows, int cols, int type);


/** Finds the Cholesky decomposition.
  *
  * \param src Symmetric positive-definite matrix
  * \return Lower traingular matrix L, such as L * L.t() == src,
            or empty matrix if the decomposition doesn't exist
  */
cv::Mat DecomposeCholesky(cv::InputArray src);


/** Finds a decomposition of a matrix into the product of an upper triangular and its transpose.
  *
  * \param src Symmetric positive-definite matrix
  * \return Upper traingular matrix U, such as U * U.t() == src,
            or empty matrix if the decomposition doesn't exist
  */
cv::Mat DecomposeUUt(cv::InputArray src);


/** Truncs eigenvalues of a symmetric matrix.
  *
  * \param src Symmetric matrix
  * \param interval Desired eigenvalues interval
  * \return Symmetric matrix with truncated eigenvalues
  */
cv::Mat TruncEigenvals(cv::InputArray src, Interval interval);


/** Extracts matched keypoints.
  *
  * \param f1 First image features
  * \param f2 Second image features
  * \param matches Matches vector
  * \param kps1 First image keypoints
  * \param kps2 Second image keypoints
  */
void ExtractMatchedKeypoints(const cv::detail::ImageFeatures &f1,
                             const cv::detail::ImageFeatures &f2,
                             const std::vector<cv::DMatch> &matches,
                             cv::OutputArray kps1, cv::OutputArray kps2);

} // namespace autocalib

#include "core_inl.h"

#endif // AUTOCALIB_CORE_H_
