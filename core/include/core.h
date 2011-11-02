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
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/util.hpp>
#include <config.h>

namespace autocalib {

    typedef std::map<int, cv::Ptr<cv::detail::ImageFeatures> > FeaturesCollection;
    typedef std::map<std::pair<int, int>, cv::Ptr<std::vector<cv::DMatch> > > MatchesCollection;
    
    /** 2D projective space homographies collection. */
    typedef std::map<std::pair<int, int>, cv::Mat> HomographiesP2;
    
    typedef std::map<std::pair<int, int>, cv::Mat> RelativeRotationMats;
    typedef std::map<std::pair<int, int>, double> RelativeConfidences;
    typedef std::map<int, cv::Mat> AbsoluteRotationMats;

    /** Describes a motion data. */
    struct Motion {
        Motion() {}
        Motion(cv::Mat R, cv::Mat T) : R(R), T(T) {}

        cv::Mat R;
        cv::Mat T;
    };

    typedef std::map<int, Motion> AbsoluteMotions;

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
        static RigidCamera LocalToWorld(const cv::Mat &K, const cv::Mat &R, const cv::Mat &center) {
            cv::Mat R_inv = R.inv();
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
        REFINE_FLAG_FX = 1,
        REFINE_FLAG_FY = 2,
        REFINE_FLAG_PPX = 4,
        REFINE_FLAG_PPY = 8,
        REFINE_FLAG_SKEW = 16,
        REFINE_FLAG_ALL = 31
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
                             int params_to_refine = REFINE_FLAG_ALL);   


    /** Refines a stereo camera paramers.
      *
      * The following notation is used for frames (or cameras) indices: (2*i, 2*i+1) is
      * the indices pair of two frames in the i'th stereo pair. 2*i is the left frame,
      * 2*i+1 is the right frame.
      *
      * \param cam Stereo camera parameters
      * \param motions Absolute motions (R,T) of stereo pairs
      * \param features Frames features
      * \param matches Matches between left frames of stereo pairs and between
                       left and right frames of stereo pairs
      * \return Epipolar error
      */
    double RefineStereoCamera(RigidCamera &cam, AbsoluteMotions motions,
                              const FeaturesCollection &features, const MatchesCollection &matches);


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

        /** Creates a features finder.
          *
          * \return Pointer to features finder object
          */
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


    class BestOf2NearestMatcherCreator {
    public:
        BestOf2NearestMatcherCreator()
            : matcher(new cv::FlannBasedMatcher()), match_conf(0.65f) {}

        cv::Ptr<cv::detail::FeaturesMatcher> Create() {
            return new BestOf2NearestMatcher(matcher, match_conf);
        }

        cv::Ptr<cv::DescriptorMatcher> matcher;
        float match_conf;
    };


    //============================================================================
    // Structure and motion

    /** Extracts the second camera matrix from the fundamental matrix.
      *
      * See details in Hartey R., Zisserman A., "Multiple View Geometry", 2nd ed., p. 256.
      *
      * \param F Fundamental matrix
      * \return Camera matrix for the second image in pair
      */
    cv::Mat Extract2ndCameraMatFromF(cv::InputArray F);


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
        virtual void triangulate(const IProjectiveCamera &P1, const IProjectiveCamera &P2, 
                                 cv::InputArray xy1, cv::InputArray xy2, 
                                 cv::InputOutputArray xyzw) = 0;
    };


    /** DLT (homogeneous) triangulation method. 
      *
      * See details in Hartey R., Zisserman A., "Multiple View Geometry", 2nd ed., p. 312.
      */
    class DltTriangulation : public ITriangulationMethod {
    public:
        virtual void triangulate(const IProjectiveCamera &P1, const IProjectiveCamera &P2, 
                                 cv::InputArray xy1, cv::InputArray xy2, 
                                 cv::InputOutputArray xyzw);
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
      * \param F Fundamental matrix, such as p1' * F * p2 = 0
      * \return RMS point-to-epipolar distance
      */
    double CalcRmsEpipolarDistance(cv::InputArray xy1, cv::InputArray xy2, cv::InputArray F);


    /** Finds a 3D projective space homography linearly.
      *
      * The algorithm is sensitive to outliers and it requires 5 points as minimum.
      *
      * \param xyzw1 First point cloud
      * \param xyzw2 Second point cloud
      * \return 3D projective sapce homography mapping xyzw1 into xyzw2
      */
    cv::Mat FindHomographyLinear(cv::InputArray xyzw1, cv::InputArray xyzw2);


    /** Calculates a plane-at-infinity coordinates from a homography.
      *
      * See details in Hartey R., Zisserman A., "Multiple View Geometry", 2nd ed., p. 495.
      *
      * \param H Homography
      * \return Plane-at-infinity coordinates (4x1 vector)
      */
    cv::Mat CalcPlaneAtInfinity(cv::InputArray H);


    //============================================================================
    // Other

    inline double Sqr(double x) { return x * x; }   


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
      * \param graph Efficient correspondences subgraph (it's an oriented graph and it's a tree)
      * \param rel_confs_eff Efficient pairwise matches confiedences (optional)
      * \return Extracted graph center (if many then one of)
      */
    int ExtractEfficientCorrespondences(int num_frames, const RelativeConfidences &rel_confs,
                                        cv::detail::Graph &eff_corresp, RelativeConfidences *rel_confs_eff = 0);


    /** Computes absolute rotation matrices from relative ones according to the
      * efficient correspondeces subgraph.
      *
      * \param rel_rmats Pairwise rotations
      * \param eff_corresp Efficient correspondeces subgraph
      * \param ref_frame_idx Reference frame index
      * \param abs_rmats Absolute rotations
      */
    void CalcAbsoluteRotations(const RelativeRotationMats &rel_rmats, const cv::detail::Graph &eff_corresp,
                               int ref_frame_idx, AbsoluteRotationMats &abs_rmats);   


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

} // namespace autocalib

#include "core_inl.h"

#endif // AUTOCALIB_CORE_H_
