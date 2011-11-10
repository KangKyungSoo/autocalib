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
    // Mono camera viewer
    // TODO use pimpl to remove the dependecy on glfw.h

    namespace internal {
        void GLFWCALL MonoViewerKeyCallback(int key, int state);
        void GLFWCALL MonoViewerWindowSizeCallback(int width, int height);
    } // namespace internal


    class MonoViewer;
    MonoViewer& mono_viewer();


    class MonoViewer {
    public:
        cv::Ptr<PointCloudScene> scene() const { return scene_; }
        void set_scene(cv::Ptr<PointCloudScene> scene) { scene_ = scene; }

        RigidCamera camera() const { return camera_; }
        void set_camera(RigidCamera camera) { camera_ = camera; }

        cv::Rect view_port() const { return view_port_; }
        void set_view_port(cv::Rect view_port) { view_port_ = view_port; }

        cv::Size window_size() const { return window_size_; }
        void set_window_size(cv::Size window_size) { window_size_= window_size; }

        void set_camera_snapshots_output(std::vector<RigidCamera> *cameras) { cameras_ = cameras; }
        void set_feature_snapshots_output(FeaturesCollection *features) { features_ = features; }

        void Run();

    private:
        friend void GLFWCALL internal::MonoViewerKeyCallback(int key, int state);
        friend void GLFWCALL internal::MonoViewerWindowSizeCallback(int width, int height);
        friend MonoViewer& mono_viewer();

        MonoViewer() : cameras_(0), features_(0) {
            cv::Mat_<double> K = cv::Mat::eye(3, 3, CV_64F);
            K(0, 0) = 3000; K(0, 2) = 960;
            K(1, 1) = 3000; K(1, 2) = 540;
            set_camera(RigidCamera(K, cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F)));
            set_view_port(cv::Rect(0, 0, 1920, 1080));
            set_window_size(cv::Size(640, 360));
        }

        cv::Ptr<PointCloudScene> scene_;
        RigidCamera camera_;
        cv::Rect view_port_;
        cv::Size window_size_;
        std::vector<RigidCamera> *cameras_;
        FeaturesCollection *features_;

        bool running_;
    };

} // namespace evaluation
} // namespace autocalib

#endif // AUTOCALIB_EVALUATION_H_
