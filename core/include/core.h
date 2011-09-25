#ifndef AUTOCALIB_CORE_H_
#define AUTOCALIB_CORE_H_

#include <vector>
#include <map>
#include <utility>
#include <opencv2/core/core.hpp>
#include <opencv2/stitching/detail/matchers.hpp>

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
        VerboseNo = 0,
        VerboseSummary = 1,
        VerboseIter = 2
    };

    /** \param crit Termination criteria
      * \param verbose Verbosity level
      * \see Verbose
      */
    MinimizeOpts(cv::TermCriteria crit = DefaultCrit(), Verbose verbose = VerboseNo) {
        Init(crit, verbose);
    }

    /** Constructs options using the default termination criteria.
      *
      * \param verbose Verbosity level
      * \see Verbose
      */
    MinimizeOpts(Verbose verbose) { Init(DefaultCrit(), verbose); }

    /** \return Default termination criteria. */
    static cv::TermCriteria DefaultCrit() {
        return cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                                100, 1e-3);
    }

    const Verbose verbose() const { return verbose_; }
    const cv::TermCriteria& crit() const { return crit_; }

private:
    void Init(cv::TermCriteria term_crit, Verbose verbose) {
        crit_ = term_crit;
        verbose_ = verbose;
    }

    Verbose verbose_;
    cv::TermCriteria crit_;
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

/** Calculates rotational camera intrinsics using linear algorithm.
  *
  * See details in Hartey R., Zisserman A., "Multiple View Geometry", 2nd ed., p. 482.
  *
  * \param Hs Projective plane homographies (64F)
  * \return Camera intrinsics (64F)
  */
cv::Mat CalibRotationalCameraLinear(cv::InputArrayOfArrays Hs);


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


/** Performs Cholesky decomposition.
  *
  * \param src Symmetric positive-definite matrix
  * \return Lower traingular matrix L, such as L * L.t() == src,
            or empty matrix if decomposition doesn't exist
  */
cv::Mat DecomposeCholesky(cv::InputArray src);


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
