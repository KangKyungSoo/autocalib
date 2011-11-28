#ifndef AUTOCALIB_EVALUATION_H_
#define AUTOCALIB_EVALUATION_H_

#include <vector>
#include <GL/glfw.h>
#include <opencv2/core/core.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <core/include/core.h>

namespace autocalib {
namespace evaluation {

    //============================================================================
    // Scenes base classes

    /** Synthetic scene base class. */
    class ISyntheticScene {
    public:
        virtual ~ISyntheticScene() {}

        /** Takes a shot of the scene.
          *
          * \param camera Rigid camera parameters
          * \param viewport Viewing region
          * \param features Result image features
          */
        virtual void TakeShot(const RigidCamera &camera, cv::Rect viewport,
                              cv::detail::ImageFeatures &features) = 0;
    };


    /** Describes a point cloud. */
    class PointCloud {
    public:

        /** \return Number of points */
        virtual int pointCount() const { return static_cast<int>(points_.size()); }

        /** \return 3D point local coordinates */
        cv::Point3d localPointAt(int idx) const { return points_[idx]; }

    protected:
        std::vector<cv::Point3d> points_;
    };


    /** Describes a synthetic point cloud scene. */
    class PointCloudScene : public ISyntheticScene, public PointCloud {
    public:
        virtual ~PointCloudScene() {}

        /** Takes a shot of the scene.
          *
          * \param camera Rigid camera parameters
          * \param viewport Viewing region
          * \param features Result image features
          */
        virtual void TakeShot(const RigidCamera &camera, cv::Rect viewport,
                              cv::detail::ImageFeatures &features);

        /** \return Local to world coordinates rotation matrix */
        const cv::Mat R() const { return R_; }

        /** \param R Local to world coordinates rotation matrix */
        void set_R(const cv::Mat &R) {
            CV_Assert(R.size() == cv::Size(3, 3) && R.type() == CV_64F);
            R_ = R.clone();
        }

        /** \return Local to world coordinates translation vector */
        const cv::Mat T() const { return T_; }

        /** \param T Local to world coordinates translation vector */
        void set_T(const cv::Mat &T) {
            CV_Assert(T.size() == cv::Size(1, 3) && T.type() == CV_64F);
            T_ = T.clone();
        }

    protected:

        /** Constructs a scene without a transformation. */
        PointCloudScene() {
            set_R(cv::Mat::eye(3, 3, CV_64F));
            set_T(cv::Mat::zeros(3, 1, CV_64F));
        }

        /** Checks point visibility.
          *
          * \return true if the point is visible from the given origin, false otherwise */
        virtual bool IsVisible(const cv::Point3d &point, const cv::Point3d &origin) const = 0;

        cv::Mat_<double> R_, T_;
    };


    /** Synthetic scenes factory. */
    class IPointCloudSceneCreator {
    public:
        virtual ~IPointCloudSceneCreator() {}

        /** Creates a synthetic scene.
          *
          * \param num_points Number of points
          * \param rng Pseudo random number generator
          */
        virtual cv::Ptr<PointCloudScene> Create(int num_points, cv::RNG &rng) const = 0;
    };


    //============================================================================
    // Concrete scenes

    /** Describes a synthetic sphere scene.
      *
      * Created sphere has unit radius and center in the origin.
      */
    class SphereScene : public PointCloudScene {
    public:

        /** Creates a sphere scene.
          *
          * \param num_points Number of points on the sphere
          * \param rng Pseudo random number generator
          */
        SphereScene(int num_points, cv::RNG &rng);

    private:
        virtual bool IsVisible(const cv::Point3d &point, const cv::Point3d &origin) const;
    };


    class SphereSceneCreator : public IPointCloudSceneCreator {
    public:
        virtual cv::Ptr<PointCloudScene> Create(int num_points, cv::RNG &rng) const {
            return new SphereScene(num_points, rng);
        }
    };


    /** Describes a synthetic cube scene.
      *
      * Creates cube has unit edge length and center in the origin.
      */
    class CubeScene : public PointCloudScene {
    public:

        /** Creates a cube scene.
          *
          * \param num_points Number of points on the cube
          * \param rng Pseudo random number generator
          */
        CubeScene(int num_points, cv::RNG &rng);

    private:
        virtual bool IsVisible(const cv::Point3d &point, const cv::Point3d &origin) const;
    };


    class CubeSceneCreator : public IPointCloudSceneCreator {
    public:
        virtual cv::Ptr<PointCloudScene> Create(int num_points, cv::RNG &rng) const {
            return new CubeScene(num_points, rng);
        }
    };


    class CompositeSceneBuilder;

    /** Describes a composite synthetic scene. */
    class CompositeScene : public ISyntheticScene {
    public:
        typedef std::vector<cv::Ptr<PointCloudScene> > ScenesCollection;

        virtual void TakeShot(const RigidCamera &camera, cv::Rect viewport,
                              cv::detail::ImageFeatures &features);

    private:
        ScenesCollection scenes_;

        friend class CompositeSceneBuilder;
    };


    class CompositeSceneBuilder {
    public:

        /** Adds a scene.
          *
          * \param scene Synthetic scene
          */
        void Add(cv::Ptr<PointCloudScene> scene) {
            scenes_.push_back(scene);
        }

        /** \return Composite scene */
        cv::Ptr<CompositeScene> Build() {
            CompositeScene *result = new CompositeScene();
            result->scenes_ = scenes_;
            return result;
        }

    private:
        CompositeScene::ScenesCollection scenes_;
    };


    //============================================================================
    // Other

    /** Matches two synthetic scene shots.
      *
      * \param f1 First shot features
      * \param f2 Second shot features
      * \param matches Found matches
      */
    void MatchSyntheticShots(const cv::detail::ImageFeatures &f1, const cv::detail::ImageFeatures &f2,
                             std::vector<cv::DMatch> &matches);

    /** Creates an image from image features.
      *
      * \param features Image features
      * \return Result image
      */
    cv::Mat CreateImage(const cv::detail::ImageFeatures &features);       


    //========================================================================
    // Camera viewers

    namespace internal {
        void GLFWCALL MonoViewerKeyCallback(int key, int state);
        void GLFWCALL MonoViewerWindowSizeCallback(int width, int height);
        void GLFWCALL MonoViewerMousePosCallback(int x, int y);
        void GLFWCALL MonoViewerMouseButtonCallback(int button, int state);
    } // namespace internal


    class MonoViewer;
    MonoViewer& the_mono_viewer();


    class MonoViewer {
    public:
        void set_scene(cv::Ptr<PointCloudScene> scene) { scene_ = scene; }

        /** \return Camera intinsics */
        cv::Mat_<double> K() const { return K_; }
        /** \param K Camera intinsics */
        void set_K(cv::Mat_<double> K) { K_ = K; }

        /** \return Local-to-world rotaton */
        cv::Mat_<double> R() const { return R_; }
        /** \param R Local-to-world rotaton */
        void set_R(cv::Mat_<double> R) { R_ = R; }

        /** \return Local-to-world translation */
        cv::Mat_<double> T() const { return T_; }
        /** \param T Local-to-world translation */
        void set_T(cv::Mat_<double> T) { T_ = T; }

        /** \param view_port (it defines the final image size) */
        void set_view_port(cv::Rect view_port) { view_port_ = view_port; }

        void set_window_size(cv::Size window_size) { window_size_= window_size; }

        void set_move_speed(double speed) { move_speed_ = speed; }
        void set_rotation_speed(double speed) { rotation_speed_ = speed; }

        void set_camera_snapshots_output(std::vector<RigidCamera> *cameras) {
            cameras_ = cameras;
        }
        void set_feature_snapshots_output(FeaturesCollection *features) {
            features_ = features;
        }

        void Run();

    private:
        MonoViewer();
        void InitRun();
        void InitOpenGl();

        cv::Ptr<PointCloudScene> scene_;
        cv::Mat_<double> K_, R_, T_;
        cv::Rect view_port_;

        cv::Size window_size_;
        bool is_running_;

        bool is_left_button_pressed_;
        int start_x_, start_y_;
        cv::Mat_<double> start_R_;
        double move_speed_;
        double rotation_speed_;

        int prev_key_;

        std::vector<RigidCamera> *cameras_;
        FeaturesCollection *features_;

        friend void GLFWCALL internal::MonoViewerKeyCallback(int key, int state);
        friend void GLFWCALL internal::MonoViewerWindowSizeCallback(int width, int height);
        friend void GLFWCALL internal::MonoViewerMousePosCallback(int x, int y);
        friend void GLFWCALL internal::MonoViewerMouseButtonCallback(int button, int state);
        friend MonoViewer& the_mono_viewer();
    };


    namespace internal {
        void GLFWCALL StereoViewerKeyCallback(int key, int state);
        void GLFWCALL StereoViewerWindowSizeCallback(int width, int height);
        void GLFWCALL StereoViewerMousePosCallback(int x, int y);
        void GLFWCALL StereoViewerMouseButtonCallback(int button, int state);
        void GLFWCALL StereoViewerMouseWheelCallback(int pos);
    } // namespace internal


    class StereoViewer;
    StereoViewer& the_stereo_viewer();


    class StereoViewer {
    public:
        void set_scene(cv::Ptr<ISyntheticScene> scene) { scene_ = scene; }

        /** \return Camera intinsics */
        cv::Mat_<double> K() const { return K_; }
        /** \param K Camera intinsics */
        void set_K(cv::Mat_<double> K) { K_ = K; }

        /** \return Local-to-world rotaton of the right camera */
        cv::Mat_<double> R() const { return R_; }
        /** \param R Local-to-world rotaton of the right camera */
        void set_R(cv::Mat_<double> R) { R_ = R; }

        /** \return Local-to-world translation of the right camera (baseline) */
        cv::Mat_<double> T() const { return T_; }
        /** \param T Local-to-world translation of the right camera (baseline) */
        void set_T(cv::Mat_<double> T) { T_ = T; }

        /** \return Local-to-world rotation of stereo pair */
        cv::Mat_<double> Rg() const { return Rg_; }
        /** \param Rg Local-to-world rotation of stereo pair */
        void set_Rg(cv::Mat_<double> Rg) { Rg_ = Rg; }

        /** \return Local-to-world translation of stereo pair */
        cv::Mat_<double> Tg() const { return Tg_; }
        /** \param Tg Local-to-world translation of stereo pair */
        void set_Tg(cv::Mat_<double> Tg) { Tg_ = Tg; }

        /** \param view_port (it defines final image size) */
        void set_view_port(cv::Rect view_port) { view_port_ = view_port; }

        void set_window_size(cv::Size window_size) { window_size_= window_size; }

        void set_move_speed(double speed) { move_speed_ = speed; }
        void set_rotation_speed(double speed) { rotation_speed_ = speed; }
        void set_baseline_speed(double speed) { baseline_speed_ = speed; }

        void set_left_camera_snapshots_output(std::vector<RigidCamera> *cameras) {
            left_cameras_ = cameras;
        }
        void set_right_camera_snapshots_output(std::vector<RigidCamera> *cameras) {
            right_cameras_ = cameras;
        }

        void set_left_feature_snapshots_output(FeaturesCollection *features) {
            left_features_ = features;
        }
        void set_right_feature_snapshots_output(FeaturesCollection *features) {
            right_features_ = features;
        }

        void Run();

    private:
        StereoViewer();
        void InitRun();
        void InitOpenGl();

        cv::Ptr<ISyntheticScene> scene_;
        cv::Mat_<double> K_, R_, T_, Rg_, Tg_;
        cv::Rect view_port_;

        cv::Size window_size_;
        bool is_running_;

        bool is_left_button_pressed_;
        int start_x_, start_y_;
        cv::Mat_<double> start_orientation_;

        double move_speed_;
        double rotation_speed_;
        double baseline_speed_;

        int prev_wheel_pos_;
        int prev_key_;

        enum {
            ORIENTATION_MODE_STEREO_PAIR,
            ORIENTATION_MODE_RELATIVE
        } orientation_mode_;

        std::vector<RigidCamera> *left_cameras_;
        std::vector<RigidCamera> *right_cameras_;
        FeaturesCollection *left_features_;
        FeaturesCollection *right_features_;

        friend void GLFWCALL internal::StereoViewerKeyCallback(int key, int state);
        friend void GLFWCALL internal::StereoViewerWindowSizeCallback(int width, int height);
        friend void GLFWCALL internal::StereoViewerMousePosCallback(int x, int y);
        friend void GLFWCALL internal::StereoViewerMouseButtonCallback(int button, int state);
        friend void GLFWCALL internal::StereoViewerMouseWheelCallback(int pos);
        friend StereoViewer& the_stereo_viewer();
    };


    //========================================================================
    // Manual features extraction

    namespace internal {
        void GLFWCALL KeypointsExtractorKeyCallback(int key, int state);
        void GLFWCALL KeypointsExtractorWindowSizeCallback(int width, int height);
        void GLFWCALL KeypointsExtractorMouseButtonCallback(int button, int state);
    } // namespace internal


    class KeypointsExtractor;
    KeypointsExtractor& the_keypoints_extractor();


    class KeypointsExtractor {
    public:
        void set_image(cv::Mat image) {
            CV_Assert(image.type() == CV_8UC3);
            image_ = image;
        }

        void set_window_size(cv::Size window_size) { window_size_= window_size; }

        void set_keypoints_output(std::vector<cv::Point2f> *keypoints) {
            keypoints_ = keypoints;
        }

        void Run();

    private:
        KeypointsExtractor();
        void InitOpenGl();
        void InitRun();

        cv::Size window_size_;
        bool is_running_;
        GLuint texture_;

        cv::Mat image_;
        std::vector<cv::Point2f> *keypoints_;

        friend void GLFWCALL internal::KeypointsExtractorKeyCallback(int key, int state);
        friend void GLFWCALL internal::KeypointsExtractorWindowSizeCallback(int width, int height);
        friend void GLFWCALL internal::KeypointsExtractorMouseButtonCallback(int button, int state);
        friend KeypointsExtractor& the_keypoints_extractor();
    };


    //========================================================================
    // Manual features matching

    namespace internal {
        void GLFWCALL FeaturesMatcherKeyCallback(int key, int state);
        void GLFWCALL FeaturesMatcherWindowSizeCallback(int width, int height);
        void GLFWCALL FeaturesMatcherMouseButtonCallback(int button, int state);
    } // namespace internal


    class FeaturesMatcher;
    FeaturesMatcher& the_features_matcher();


    class FeaturesMatcher {
    public:
        void set_1st_image(cv::Mat image, const std::vector<cv::Point2f> &keypoints) {
            CV_Assert(image.type() == CV_8UC3);
            image1_ = image;
            keypoints1_ = &keypoints;
        }

        void set_2nd_image(cv::Mat image, const std::vector<cv::Point2f> &keypoints) {
            CV_Assert(image.type() == CV_8UC3);
            image2_ = image;
            keypoints2_ = &keypoints;
        }

        void set_window_size(cv::Size window_size) { window_size_= window_size; }

        void set_matches_output(std::vector<cv::DMatch> *matches) {
            matches_ = matches;
        }

        void Run();

    private:
        FeaturesMatcher();
        void InitOpenGl();
        void InitRun();
        void ScreenToLocal(int x, int y, float &u, float &v);

        cv::Size window_size_;
        bool is_running_;
        GLuint texture1_, texture2_;

        cv::Mat image1_, image2_;
        const std::vector<cv::Point2f> *keypoints1_, *keypoints2_;
        std::vector<cv::DMatch> *matches_;

        bool is_keypoint1_selected_;
        bool is_keypoint2_selected_;
        int prev_keypoint_index_;       

        friend void GLFWCALL internal::FeaturesMatcherKeyCallback(int key, int state);
        friend void GLFWCALL internal::FeaturesMatcherWindowSizeCallback(int width, int height);
        friend void GLFWCALL internal::FeaturesMatcherMouseButtonCallback(int button, int state);
        friend FeaturesMatcher& the_features_matcher();
    };

} // namespace evaluation
} // namespace autocalib

#endif // AUTOCALIB_EVALUATION_H_
