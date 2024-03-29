#ifndef AUTOCALIB_CORE_H_
#define AUTOCALIB_CORE_H_

#include <vector>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <limits>
#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/util.hpp>
#include <config.h>

namespace autocalib {

typedef std::map<int, cv::Ptr<cv::detail::ImageFeatures> > FeaturesCollection;

typedef std::map<std::pair<int, int>, cv::Ptr<std::vector<cv::DMatch> > > MatchesCollection;

/** 2D projective space homographies collection. */
typedef std::map<std::pair<int, int>, cv::Mat> HomographiesP2;

/** 3D projective space homographies collection. */
typedef std::map<std::pair<int, int>, cv::Mat> HomographiesP3;

typedef std::map<std::pair<int, int>, cv::Mat> RelativeRotationMats;

typedef std::map<std::pair<int, int>, double> RelativeConfidences;

typedef std::map<int, cv::Mat> AbsoluteRotationMats;

class Motion;

typedef std::map<std::pair<int, int>, Motion> RelativeMotions;

typedef std::map<int, Motion> AbsoluteMotions;


//============================================================================
// Cameras and motions

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
    cv::Mat_<double> P_;
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
    static RigidCamera FromLocalToWorld(const cv::Mat &K, const cv::Mat &R, const cv::Mat &center) {
        cv::Mat R_inv = R.t();
        return RigidCamera(K, R_inv, -R_inv * center);
    }

    /** Creates a rigid camera from a projective camera matrix.
      *
      * \param P Projective camera matrix
      * \return Rigid camera
      */
    static RigidCamera FromProjectiveMat(const cv::Mat &P);

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

    /** \return Camera intrinsics */
    const cv::Mat& K() const { return K_; }

    /** \return World to local coordinates rotation matrix */
    const cv::Mat& R() const { return R_; }

    /** \return World to local coordinates translation vector */
    const cv::Mat& T() const { return T_; }

private:
    cv::Mat_<double> K_, R_, T_;
};


/** Describes a rigid motion. */
class Motion {
public:

    /** \return Identity motion object */
    Motion() {
        set_R(cv::Mat::eye(3, 3, CV_64F));
        set_T(cv::Mat::zeros(3, 1, CV_64F));
    }

    /** \param R Rotation matrix
      * \param T Translation vector
      * \return Motion object
      */
    Motion(const cv::Mat &R, const cv::Mat &T) {
        set_R(R);
        set_T(T);
    }

    /** \return Rotation matrix */
    const cv::Mat& R() const { return R_; }

    /** \param R Rotation matrix */
    void set_R(const cv::Mat &R) {
        CV_Assert(R.type() == CV_64F && R.size() == cv::Size(3, 3));
        R_ = R.clone();
    }

    /** \return Translation matrix */
    const cv::Mat& T() const { return T_; }

    /** \param T Translation vector */
    void set_T(const cv::Mat &T) {
        CV_Assert(T.type() == CV_64F && T.size() == cv::Size(1, 3));
        T_ = T.clone();
    }

private:
    cv::Mat_<double> R_, T_;
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
// Rotation model camera autocalibration

/** Calculates rotational camera intrinsics using a linear algorithm.
  *
  * See details in Hartey R., Zisserman A., "Multiple View Geometry", 2nd ed., p. 482.
  *
  * \param Hs Projective plane homographies
  * \param residual_error Relative residual error (optional)
  * \return Camera intrinsics
  */
cv::Mat CalibRotationalCameraLinear(const HomographiesP2 &Hs, double *residual_error = 0);


/** Calculates rotational camera intrinsics using a linear algorithm with the zero skew assumption.
  *
  * See details in Hartey R., Zisserman A., "Multiple View Geometry", 2nd ed., p. 482.
  *
  * \param Hs Projective plane homographies
  * \param residual_error Relative residual error (optional)
  * \return Camera intrinsics, where skew is zero
  */
cv::Mat CalibRotationalCameraLinearNoSkew(const HomographiesP2 &Hs, double *residual_error = 0);


/** Rigid camera refinement method flags. */
enum RefineFlag {
    REFINE_FLAG_K_FX = 1,
    REFINE_FLAG_K_FY = 2,
    REFINE_FLAG_K_PPX = 4,
    REFINE_FLAG_K_PPY = 8,
    REFINE_FLAG_K_SKEW = 16,
    REFINE_FLAG_K_ALL = 31
};


/** Refines a rigid camera parameters by minimizing overall reprojection error.
  *
  * \param K Camera intrinsics
  * \param Rs Camera rotations
  * \param features Features collection
  * \param matches Matches collection
  * \param params_to_refine Flags indicating parameters which should be refined
  * \return Reprojection RMS error
  * \see RefineFlag
  */
double RefineRigidCamera(cv::InputOutputArray K, AbsoluteRotationMats Rs,
                         const FeaturesCollection &features, const MatchesCollection &matches,
                         int params_to_refine = REFINE_FLAG_K_ALL);


//============================================================================
// Stereo camera autocalibration

/** \return true if both frames are left and right frames from the same stereo pair, false otherwise */
inline bool IsLeftRightPair(int index1, int index2) {
    return index1 / 2 == index2 / 2 && index2 == index1 + 1;
}


/** \return true if both frame are left frames of different stereo pairs, false otherwise */
inline bool BothAreLeft(int index1, int index2) {
    return index1 % 2 == 0 && index2 % 2 == 0 && index1 != index2;
}


/** Reconstructs point clouds in the P3 space.
  *
  * See details in Hartey R., Zisserman A., "Multiple View Geometry", 2nd ed., p. 496.
  * When the function finishes, keypoint arrays will contain images of points shared amongst the both pairs.
  *
  * \param P_l Left camera projective matrix (the same for both pairs)
  * \param P_r Right camera projective matrix (the same for both pairs)
  * \param xy_l0 First pair left image keypoints
  * \param xy_r0 First pair right image keypoints
  * \param xy_l1 Second pair left image keypoints
  * \param xy_r1 Second pair right image keypoints
  * \param matches_lr0 Matches between left and right images of the first pair
  * \param matches_lr1 Matches between left and right images of the second pair
  * \param matches_ll Matches between left images of two pairs
  * \return Two point clouds: i'th point of the 1st cloud corresponds to the i'th point of the 2nd cloud.
            Number of points are the same.
  */
std::pair<cv::Mat, cv::Mat> ReconstructPointClouds(
    cv::InputOutputArray P_l, cv::InputOutputArray P_r,
    cv::InputOutputArray xy_l0, cv::InputOutputArray xy_r0,
    cv::InputOutputArray xy_l1, cv::InputOutputArray xy_r1,
    const cv::Ptr<std::vector<cv::DMatch> > &matches_lr0,
    const cv::Ptr<std::vector<cv::DMatch> > &matches_lr1,
    const cv::Ptr<std::vector<cv::DMatch> > &matches_ll);


/** Upgrades a projective point cloud to the affine one.
  *
  * \param pinf Plane-at-infinity
  * \param xyzw Point cloud
  */
void UpgradeProjectiveToAffine(cv::InputArray pinf, cv::InputOutputArray xyzw);


// TODO add docs
void AffineRectify(
        cv::InputArray pinf,
        cv::InputOutputArray P_l, cv::InputOutputArray P_r, cv::InputOutputArray H01,
        int num_points, cv::InputOutputArray xyzw0, cv::InputOutputArray xyzw1);


/** Performs affine rectification of the stereo pair by two image pairs.
  *
  * See details in Hartey R., Zisserman A., "Multiple View Geometry", 2nd ed., p. 496.
  * When the function finishes, keypoint arrays will contain images of points shared amongst the both pairs.
  *
  * \param P_l Left camera projective matrix (the same for both pairs)
  * \param P_r Right camera projective matrix (the same for both pairs)
  * \param xy_l0 First pair left image keypoints
  * \param xy_r0 First pair right image keypoints
  * \param xy_l1 Second pair left image keypoints
  * \param xy_r1 Second pair right image keypoints
  * \param matches_lr0 Matches between left and right images of the first pair
  * \param matches_lr1 Matches between left and right images of the second pair
  * \param matches_ll Matches between left images of two pairs
  * \param H01 Homography mapping the first pair point cloud into the second pair point cloud
  * \param num_iters Number of iterations for H estimation
  * \param subset_size Subset size for H estimation
  * \param thresh Error threshold of H estimation
  * \param xyzw0 First pair point cloud
  * \param xyzw1 Second pair point cloud
  * \return true if it succeded, false otherwise
  */
bool AffineRectifyStereoCameraByTwoShots(
        cv::InputOutputArray P_l, cv::InputOutputArray P_r,
        cv::InputOutputArray xy_l0, cv::InputOutputArray xy_r0,
        cv::InputOutputArray xy_l1, cv::InputOutputArray xy_r1,
        const cv::Ptr<std::vector<cv::DMatch> > &matches_lr0, const cv::Ptr<std::vector<cv::DMatch> > &matches_lr1,
        const cv::Ptr<std::vector<cv::DMatch> > &matches_ll,
        int num_iters, int subset_size, double thresh,
        cv::OutputArray H01, cv::OutputArray xyzw0, cv::OutputArray xyzw1);


/** Computes the symmetric point-to-epipolar distance.
  *
  * See details in Hartey R., Zisserman A., "Multiple View Geometry", 2nd ed., p. 287
  */
double SymEpipDist2(double x1, double y1, const cv::Mat F12, double x2, double y2);


/** Refines a stereo camera parameters.
  *
  * \param cam Stereo camera parameters
  * \param features Frames features
  * \param matches Matches between left frames of stereo pairs and between
                   left and right frames of stereo pairs
  * \param params_to_refine Flags indicating parameters which should be refined
  * \return Epipolar distance error
  * \see RefineFlag
  */
double RefineStereoCamera(RigidCamera &cam, const FeaturesCollection &features,
                          const MatchesCollection &matches, int params_to_refine = REFINE_FLAG_K_ALL);


/** Refines a stereo camera parameters.
  *
  * \param cam Stereo camera parameters
  * \param motions Absolute motions (R,T) of stereo pairs
  * \param features Frames features
  * \param matches Matches between left frames of stereo pairs and between
                   left and right frames of stereo pairs
  * \param params_to_refine Flags indicating parameters which should be refined
  * \param rel_conf Matches relative confidences
  * \return Epipolar distance error
  * \see RefineFlag
  */
double RefineStereoCamera(RigidCamera &cam, AbsoluteMotions &motions,
                          const FeaturesCollection &features, const MatchesCollection &matches,
                          int params_to_refine = REFINE_FLAG_K_ALL,
                          const RelativeConfidences &rel_confs = RelativeConfidences());


//============================================================================
// Features related stuff

/** Describes an ORB features finder. */
class OrbFeaturesFinder : public cv::detail::FeaturesFinder {
public:

    /** Constructs an ORB features finder.
      *
      * \param num_features Number of desired features
      */
    OrbFeaturesFinder(int num_features) : orb_(num_features) {}

private:
    virtual void find(const cv::Mat &image, cv::detail::ImageFeatures &features) {
        orb_(image, cv::Mat(), features.keypoints, features.descriptors);
    }

    cv::ORB orb_;
};


/** Base class for features finder creators */
class FeaturesFinderCreator {
public:
    virtual ~FeaturesFinderCreator() {}
    virtual cv::Ptr<cv::detail::FeaturesFinder> Create() = 0;
};


class SurfFeaturesFinderCreator : public FeaturesFinderCreator {
public:
    SurfFeaturesFinderCreator() : hess_thresh(300), num_octaves(3), num_layers(4) {}

    virtual cv::Ptr<cv::detail::FeaturesFinder> Create() {
        return new cv::detail::SurfFeaturesFinder(hess_thresh, num_octaves, num_layers);
    }

    double hess_thresh;
    int num_octaves;
    int num_layers;
};


class OrbFeaturesFinderCreator : public FeaturesFinderCreator {
public:
    OrbFeaturesFinderCreator() : num_features(500) {}

    virtual cv::Ptr<cv::detail::FeaturesFinder> Create() {
        return new OrbFeaturesFinder(num_features);
    }

    int num_features;
};


class FeaturesMatcherCreator {
public:
    virtual ~FeaturesMatcherCreator() {}
    virtual cv::Ptr<cv::detail::FeaturesMatcher> Create() = 0;
};


class BestOf2NearestMatcher : public cv::detail::FeaturesMatcher {
public:
    BestOf2NearestMatcher(cv::Ptr<cv::DescriptorMatcher> &matcher, float match_conf)
        : matcher_(matcher), match_conf_(match_conf) {}

    virtual void match(const cv::detail::ImageFeatures &f1, const cv::detail::ImageFeatures &f2,
                       cv::detail::MatchesInfo &mi);

private:
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    float match_conf_;
};


class BestOf2NearestMatcherCreator : public FeaturesMatcherCreator {
public:
    BestOf2NearestMatcherCreator()
        : matcher(new cv::FlannBasedMatcher()), match_conf(0.65f) {}

    cv::Ptr<cv::detail::FeaturesMatcher> Create() {
        return new BestOf2NearestMatcher(matcher, match_conf);
    }

    cv::Ptr<cv::DescriptorMatcher> matcher;
    float match_conf;
};


/** Finds an assignment using the max-element method.
  *
  * \param cost Cost matrix
  * \param pairs Found pairs
  */
void FindAssignment(cv::Mat_<float> cost, std::vector<std::pair<int, int> > &pairs);


class OptAssignmentMatcher : public cv::detail::FeaturesMatcher {
public:
    virtual void match(const cv::detail::ImageFeatures &f1, const cv::detail::ImageFeatures &f2,
                       cv::detail::MatchesInfo &mi);
};


class OptAssignmentMatcherCreator : public FeaturesMatcherCreator {
public:
    cv::Ptr<cv::detail::FeaturesMatcher> Create() {
        return new OptAssignmentMatcher();
    }
};


//============================================================================
// Structure and motion

/** Extracts the camera matrix from the fundamental matrix.
  *
  * See details in Hartey R., Zisserman A., "Multiple View Geometry", 2nd ed., p. 256.
  *
  * \param F Fundamental matrix
  * \return Camera matrix for the second image in pair
  */
cv::Mat CameraMatFromFundamentalMat(cv::InputArray F);


/** Intersects matches between images in stereo pairs with matches between stereo pairs.
  *
  * The functions does assumption that matches_lr* are injective mappings from left image
  * to right image keypoints.
  *
  * \param matches_lr1 First stereo pair matches
  * \param matches_lr2 Second stereo pair matches
  * \param matches_ll Matches between left images of stereo pairs
  * \param indices Matches indices pairs vector
  */
void Intersect(const std::vector<cv::DMatch> &matches_lr1, const std::vector<cv::DMatch> &matches_lr2,
               const std::vector<cv::DMatch> &matches_ll, std::vector<std::pair<int, int> > &indices);


/** Triangulation method base class. */
class ITriangulationMethod {
public:
    virtual ~ITriangulationMethod() {}

    /** Estimates 3D projective space points coordinates from two images keypoints.
      *
      * \param P1 First camera
      * \param P2 Second camera
      * \param xy1 First image keypoints
      * \param xy2 Second image keypoints
      * \param xyzw 3D projective space points
      */
    virtual void Triangulate(const IProjectiveCamera &P1, const IProjectiveCamera &P2,
                             cv::InputArray xy1, cv::InputArray xy2,
                             cv::InputOutputArray xyzw) = 0;
};


class ITringulationMethodCreator {
public:
    virtual ~ITringulationMethodCreator() {}
    virtual cv::Ptr<ITriangulationMethod> Create() const = 0;
};


/** DLT (homogeneous) triangulation method.
  *
  * See details in Hartey R., Zisserman A., "Multiple View Geometry", 2nd ed., p. 312.
  */
class DltTriangulation : public ITriangulationMethod {
public:
    virtual void Triangulate(const IProjectiveCamera &P1, const IProjectiveCamera &P2,
                             cv::InputArray xy1, cv::InputArray xy2, cv::InputOutputArray xyzw);
};


class DltTriangulationCreator : public ITringulationMethodCreator {
public:
    virtual cv::Ptr<ITriangulationMethod> Create() const {
        return new DltTriangulation();
    }
};


/** Iterative triangulation.
  *
  * See details here: http://www.cs.unc.edu/~marc/tutorial/node68.html.
  */
class IterativeTriangulation : public ITriangulationMethod {
public:
    IterativeTriangulation() { set_num_iters(2); }

    virtual void Triangulate(const IProjectiveCamera &P1, const IProjectiveCamera &P2,
                             cv::InputArray xy1, cv::InputArray xy2, cv::InputOutputArray xyzw);

    int num_iters() const { return num_iters_; }
    void set_num_iters(int val) { num_iters_ = val; }

private:
    int num_iters_;
};


class IterativeTriangulationCreator : public ITringulationMethodCreator {
public:
    virtual cv::Ptr<ITriangulationMethod> Create() const {
        return new IterativeTriangulation();
    }
};


/** Calculates an isotropic normalization transformation matrix.
  *
  * See details in Hartey R., Zisserman A., "Multiple View Geometry", 2nd ed., p. 107.
  *
  * \param xy Image keypoints
  * \return Transformation matrix
  */
cv::Mat CalcNormalizationMat3x3(cv::InputArray xy);


/** Calculates the reprojection RMS error.
  *
  * \param xy Image keypoints
  * \param P Camera matrix
  * \param xyzw Points
  * \return RMS reprojection error
  */
double CalcRmsReprojectionError(cv::InputArray xy, cv::InputArray P, cv::InputArray xyzw);


/** Calculates the point-to-epopolar-line RMS distance.
  *
  * \param xy1 First image keypoints
  * \param xy2 Second image keypoints
  * \param F Fundamental matrix, such as p2' * F * p1 = 0
  * \param mask Inliers mask
  * \return RMS point-to-epipolar line distance
  */
double CalcRmsEpipolarDistance(cv::InputArray xy1, cv::InputArray xy2, cv::InputArray F, cv::InputArray mask = cv::noArray());


/** Finds the 3D projective space homography linearly.
  *
  * The algorithm is sensitive to outliers and it requires 5 points as minimum.
  *
  * \param xyzw1 First point cloud
  * \param xyzw2 Second point cloud
  * \return 3D projective space homography mapping xyzw1 into xyzw2
  */
cv::Mat FindHomographyP3Linear(cv::InputArray xyzw1, cv::InputArray xyzw2);


/** Finds the 3D projective space homography using MSAC procedure.
  *
  * \param xyzw1 First point cloud
  * \param xyzw2 Second point cloud
  * \param P1 First camera matrix
  * \param P2 Second camera matrix
  * \param xy_l2 First image keypoints
  * \param xy_r2 Second image keypoints
  * \param num_iters Number of iterations
  * \param subset_size Size of point subset used for estimation
  * \param err_thresh Error threshold for inliers classification
  * \return 3D projective space homography mapping xyzw1 into xyzw2
  */
cv::Mat FindHomographyP3Robust(cv::InputArray xyzw1, cv::InputArray xyzw2, cv::InputArray P1, cv::InputArray P2,
                               cv::InputArray xy_l2, cv::InputArray xy_r2, int num_iters = 100, int subset_size = 10,
                               double err_thresh = 3.0);


/** Refines 3D projective space homography.
  *
  * \param H Homography mapping xyzw cloud
  * \param xyzw Cloud to be mapped
  * \param P1 Left camera matrix (must be applied to mapped cloud)
  * \param P2 Right camera matrix (must be applied to mapped cloud)
  * \param xy1 Left image keypoints (images of mapped points)
  * \param xy2 Right image keypoints (images of mapped points)
  * \return RMS reprojection error
  */
double RefineHomographyP3(cv::InputOutputArray H, cv::InputArray xyzw, cv::InputArray P1, cv::InputArray P2,
                          cv::InputArray xy1, cv::InputArray xy2);


/** Calculates a plane-at-infinity coordinates from a homography.
  *
  * See details in Hartey R., Zisserman A., "Multiple View Geometry", 2nd ed., p. 495.
  *
  * \param H Homography (can be rescaled inside the function)
  * \return Plane-at-infinity coordinates (4x1 vector)
  */
cv::Mat CalcPlaneAtInfinity(cv::InputOutputArray H);


/** Finds the fundamental matrix from image pairs.
  *
  * \param features Features
  * \param matches Matches in (2*i, 2*i+1) or reversed format, other are ignored
  * \param thresh Error threshold
  * \param conf Confidence
  * \param method Estimation method (CV_FM_RANSAC, CV_FM_LMEDS, ...)
  * \return Fundamental matrix
  */
cv::Mat FindFundamentalMatFromPairs(const FeaturesCollection &features, const MatchesCollection &matches,
                                    int method = CV_FM_LMEDS, double thresh = 3., double conf = 0.99);


/** Finds inliers for the given fundamental matrix.
  *
  * \param f1 First frame features
  * \param f2 Second frame features
  * \param matches Matches
  * \param F Fundamental matrix
  * \param err_thresh Error threshold
  * \param mask Inliers 8U mask
  * \return Number of inliers
  */
int FindFundamentalMatInliers(const cv::detail::ImageFeatures &f1, const cv::detail::ImageFeatures &f2,
                              const std::vector<cv::DMatch> &matches, cv::InputArray F, double err_thresh,
                              cv::InputOutputArray mask);


//============================================================================
// Other

inline double sqr(double x) {
    return x * x;
}

template <typename T>
inline const T& max(const T &v1, const T &v2, const T &v3) {
    return std::max(v1, std::max(v2, v3));
}

template <typename T>
inline const T& max(const T &v1, const T &v2, const T &v3, const T &v4) {
    return std::max(v1, max(v2, v3, v4));
}


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
  * \param mat Symmetric positive-definite matrix
  * \return Lower traingular matrix L, such as L * L.t() == src,
            or empty matrix if the decomposition doesn't exist
  */
cv::Mat DecomposeCholesky(cv::InputArray mat);


/** Finds a decomposition of a matrix into the product of an upper triangular and its transpose.
  *
  * \param mat Symmetric positive-definite matrix
  * \return Upper traingular matrix U, such as U * U.t() == src,
            or empty matrix if the decomposition doesn't exist
  */
cv::Mat DecomposeUUt(cv::InputArray mat);


/** Extracts matched keypoints.
  *
  * \param f1 First image features
  * \param f2 Second image features
  * \param matches Matches vector
  * \param xy1 First image keypoints
  * \param xy2 Second image keypoints
  */
void ExtractMatchedKeypoints(const cv::detail::ImageFeatures &f1,
                             const cv::detail::ImageFeatures &f2,
                             const std::vector<cv::DMatch> &matches,
                             cv::OutputArray xy1, cv::OutputArray xy2);


/** Applies a rigid traform to a point.
  *
  * \param point Source point
  * \param R Rotation matrix
  * \param T Translation vector
  * \return Transformed point
  */
cv::Point3d TransformRigid(const cv::Point3d &point, const cv::Mat &R, const cv::Mat &T);


/** Extracts an efficient correspondences subgraph.
  *
  * \param num_frames Number of frames
  * \param rel_confs Pairwise matches confidences
  * \param eff_corresp Efficient correspondences subgraph (it's an oriented graph and it's a tree)
  * \param rel_confs_eff Efficient pairwise matches confiedences (optional)
  * \return Extracted graph center (if many then one of)
  */
int ExtractEfficientCorrespondences(int num_frames, const RelativeConfidences &rel_confs,
                                    cv::detail::Graph &eff_corresp, RelativeConfidences *rel_confs_eff = 0);


/** Computes absolute rotation matrices from the relative ones according to the
  * efficient correspondeces subgraph.
  *
  * \param rel_rmats Pairwise rotations
  * \param eff_corresp Efficient correspondeces subgraph
  * \param ref_frame_idx Reference frame index
  * \param abs_rmats Absolute rotations
  */
void CalcAbsoluteRotations(const RelativeRotationMats &rel_rmats, const cv::detail::Graph &eff_corresp,
                           int ref_frame_idx, AbsoluteRotationMats &abs_rmats);


/** Computes absolute motions from the relative ones according to the
  * efficient correspondeces subgraph.
  *
  * \param rel_motions Pairwise motions
  * \param eff_corresp Efficient correspondeces subgraph
  * \param ref_idx Reference camera index
  * \param abs_motions Absolute motions
  */
void CalcAbsoluteMotions(const RelativeMotions &rel_motions, const cv::detail::Graph &eff_corresp,
                         int ref_idx, AbsoluteMotions &abs_motions);


/** Finds an eigen decomposition of a real matrix.
  *
  * \param mat Real matrix
  * \param vals Complex row of eigenvalues
  * \param vecs Complex matrix which rows are eigenvectors
  */
void EigenDecompose(cv::InputArray mat, cv::OutputArray vals, cv::OutputArray vecs);


/** Returns skew-symmetric matrix representing cross product.
  *
  * See http://en.wikipedia.org/wiki/Cross_product.
  *
  * \param vec 3x1 vector
  * \return Cross product matrix
  */
cv::Mat CrossProductMat(cv::InputArray vec);


/** Construct the source vector by cross product matrix.
  *
  * \param mat Cross product matrix
  * \return Vector which cross product matrix equals to the given one
  */
cv::Mat VecFromCrossProductMat(cv::InputArray mat);


/** Finds the camera centre.
  *
  * \param P Camera matrix
  * \return Camera centre C such as P*C=0
  */
cv::Mat CameraCentre(cv::InputArray P);


/** Computes the Moore-Penrose pseudo-inverse of a matrix.
  *
  * \param mat Matrix
  * \return Pseudo-inverse
  */
cv::Mat PseudoInverse(cv::InputArray mat);


/** Decomposes the essential matrix into rotation and translation.
  * The solution isn't unique, see the paper for details.
  *
  * See STEREO RIG GEOMETRY DETERMINATION BY FUNDAMENTAL MATRIX DECOMPOSITION
  *     Carles Matabosch, Joaquim Salvi and Josep Forest, 2003.
  *
  * \param E Essential matrix
  * \param R Rotation matrix
  * \param T Translation vector
  */
void DecomposeEssentialMat(cv::InputArray E, cv::OutputArray R, cv::OutputArray T);

} // namespace autocalib

#include "core_inl.h"

#endif // AUTOCALIB_CORE_H_
