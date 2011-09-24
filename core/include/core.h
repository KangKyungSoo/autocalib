#ifndef AUTOCALIB_CORE_H_
#define AUTOCALIB_CORE_H_

#include <vector>
#include <map>
#include <utility>
#include <opencv2/core/core.hpp>

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

    /** Construct a rigid camera from intrinsic and extrinsic parameters.
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


/** Describes an optimization method verbosity. */
enum OptVerbose {
    OptVerboseNo,
    OptVerboseSummary,
    OptVerboseMax
};


/** Minimizes a function using the Levenberg-Marquardt algorithm.
  *
  * \param func Function to be minimized
  * \param args Function arguments
  * \param verbose Verbosity level
  * \return Found minimum value
  * \see OptVerbose
  */
template <typename Func>
double Minimize(Func func, cv::InputOutputArray args, OptVerbose verbose = OptVerboseNo);


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

} // namespace autocalib

#include "core_inl.h"

#endif // AUTOCALIB_CORE_H_
