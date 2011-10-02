#ifndef AUTOCALIB_CORE_H_
#define AUTOCALIB_CORE_H_

#include <vector>
#include <map>
#include <string>
#include <utility>
#include <limits>
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
  * \param Hs Projective plane homographies (64F)
  * \param evals_interval Interval used for DIAC eigenvalues truncation (true eigenvalues are [fx^2, fy^2, 1])
  * \return Camera intrinsics (64F)
  */
cv::Mat CalibRotationalCameraLinear(cv::InputArrayOfArrays Hs,
                                    Interval evals_interval = Interval::All());


/** Calculates rotational camera intrinsics using a linear algorithm with the zero skew assumption.
  *
  * See details in Hartey R., Zisserman A., "Multiple View Geometry", 2nd ed., p. 482.
  *
  * \param Hs Projective plane homographies (64F)
  * \param evals_interval Interval used for IAC eigenvalues truncation (true eigenvalues are [(1/fx)^2, (1/fy)^2, 1])
  * \return Camera intrinsics (64F), where skew is zero
  */
cv::Mat CalibRotationalCameraLinearNoSkew(cv::InputArrayOfArrays Hs,
                                          Interval evals_interval = Interval::All());


#if 0
/** Describes the squared error cost function (not robust). */
class SquaredCostFunc {
public:
    double operator()(double x) const { return x * x; }
    double SquareRoot(double x) const { return std::abs(x); }
};


/** Describes the Huber robust error cost function. */
class HuberCostFunction {
public:
    HuberCostFunction(double sigma) : sigma_(std::abs(sigma)),
                                      sigma_sq_(sigma * sigma) {}

    double operator()(double x) const {
        x = std::abs(x);
        return x < sigma_ ? x * x : 2 * sigma_ * x - sigma_sq_;
    }

    double SquareRoot(double x) const { return std::sqrt((*this)(x)); }

private:
    double sigma_, sigma_sq_;
};


/** Describes the Blake-Zisserman robust error cost function. */
class BlakeZissermanCostFunc {
public:
    BlakeZissermanCostFunc(double sigma) : sigma_(std::abs(sigma)),
                                           sigma_sq_(sigma * sigma) {}

    double operator()(double x) const { return std::min(x * x, sigma_sq_); }
    double SquareRoot(double x) const { return std::min(std::abs(x), sigma_); }

private:
    double sigma_, sigma_sq_;
};
#endif


/** Rigid camera refinement method flags. */
enum RefineFlag {
    REFINE_FLAG_FX = 1,
    REFINE_FLAG_FY = 2,
    REFINE_FLAG_PPX = 4,
    REFINE_FLAG_PPY = 8,
    REFINE_FLAG_SKEW = 16,
    REFINE_FLAG_ALL = 31
};


/** Refines rigid camera parameters by minimizing overal reprojection error.
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
