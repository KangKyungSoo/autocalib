#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <sstream>
#include <iostream>
#include <GL/glut.h>
#include <opencv2/features2d/features2d.hpp>
#include <include/evaluation.h>

using namespace std;
using namespace cv;

namespace autocalib {
namespace evaluation {

    void PointCloudScene::TakeShot(const RigidCamera &camera, Rect viewport,
                                   detail::ImageFeatures &features)
    {
        Mat R_inv = camera.R().inv();
        Point3d origin = Mat(-R_inv * camera.T()).at<Point3d>(0, 0);

        vector<int> visible_points;
        features.keypoints.clear();

        Mat_<double> P = camera.P();
        for (size_t i = 0; i < points_.size(); ++i) {
            Point3d pt = points_[i];
            if (IsVisible(pt, origin)) {
                Point3d pt_ = TransformRigid(pt, R_, T_);
                pt.x = P(0, 0) * pt_.x + P(0, 1) * pt_.y + P(0, 2) * pt_.z + P(0, 3);
                pt.y = P(1, 0) * pt_.x + P(1, 1) * pt_.y + P(1, 2) * pt_.z + P(1, 3);
                pt.z = P(2, 0) * pt_.x + P(2, 1) * pt_.y + P(2, 2) * pt_.z + P(2, 3);
                if (pt.z > 0) {
                    Point2f kp(float(pt.x / pt.z), float(pt.y / pt.z));
                    if (kp.x > (float)viewport.x && kp.x < float(viewport.width - 1) &&
                        kp.y > (float)viewport.y && kp.y < float(viewport.height - 1))
                    {
                        visible_points.push_back(i);
                        features.keypoints.push_back(KeyPoint(kp, 1.f));
                    }
                }
            }
        }

        features.descriptors.create(visible_points.size(), 1, CV_32S);
        for (size_t i = 0; i < visible_points.size(); ++i)
            features.descriptors.at<int>(i, 0) = visible_points[i];

        features.img_size = viewport.size();
    }


    void CompositeScene::TakeShot(const RigidCamera &camera, Rect viewport,
                                  detail::ImageFeatures &features)
    {
        vector<detail::ImageFeatures> all_features(scenes_.size());
        size_t total_num_keypoints = 0;

        for (size_t i = 0; i < scenes_.size(); ++i) {
            scenes_[i]->TakeShot(camera, viewport, all_features[i]);
            total_num_keypoints += all_features[i].keypoints.size();
        }

        features.img_size = viewport.size();
        features.keypoints.resize(total_num_keypoints);
        features.descriptors.create(total_num_keypoints, 1, CV_32S);

        int keypoint_idx = 0;
        int descr_offset = 0;
        for (size_t i = 0; i < all_features.size(); ++i) {
            for (size_t j = 0; j < all_features[i].keypoints.size(); ++j) {
                features.keypoints[keypoint_idx] = all_features[i].keypoints[j];
                features.descriptors.at<int>(keypoint_idx, 0) =
                        all_features[i].descriptors.at<int>(j, 0) + descr_offset;
                keypoint_idx++;
            }
            descr_offset += scenes_[i]->pointCount();
        }
    }


    SphereScene::SphereScene(int num_points, RNG &rng) {
        points_.resize(num_points);
        for (int i = 0; i < num_points; ++i) {
            double phi = (double)rng * 2. * CV_PI;
            double psi = (double)rng * CV_PI;
            points_[i].x = cos(phi) * sin(psi);
            points_[i].y = sin(phi) * sin(psi);
            points_[i].z = cos(psi);
        }
    }


    bool SphereScene::IsVisible(const Point3d &point, const Point3d &origin) const {
        Point3d origin_ = TransformRigid(origin, R_.t(), -R_.t() * T_);
        return point.x * origin_.x + point.y * origin_.y + point.z * origin_.z > 0;
    }


    CubeScene::CubeScene(int num_points, RNG &rng) {
        points_.resize(num_points);
        for (int i = 0; i < num_points; ++i) {
            int j = abs((int)rng) % 3;
            points_[i].x = (j == 0 ? abs((int)rng) % 2 : (double)rng) - 0.5;
            points_[i].y = (j == 1 ? abs((int)rng) % 2 : (double)rng) - 0.5;
            points_[i].z = (j == 2 ? abs((int)rng) % 2 : (double)rng) - 0.5;
        }
    }


    bool CubeScene::IsVisible(const Point3d &point, const Point3d &origin) const {
        Point3d origin_ = TransformRigid(origin, R_.t(), -R_.t() * T_);
        Point3d dir = point - origin_;
        double dist = sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        dir *= 1 / dist;

        double x, y, z, t;

        t = (-0.5 - origin_.x) / dir.x;
        y = origin_.y + t * dir.y;
        z = origin_.z + t * dir.z;
        if (y > -0.5 && y < 0.5 && z > -0.5 && z < 0.5 && t > 0 && t < dist - 1e-6)
            return false;

        t = (0.5 - origin_.x) / dir.x;
        y = origin_.y + t * dir.y;
        z = origin_.z + t * dir.z;
        if (y > -0.5 && y < 0.5 && z > -0.5 && z < 0.5 && t > 0 && t < dist - 1e-6)
            return false;

        t = (-0.5 - origin_.y) / dir.y;
        x = origin_.x + t * dir.x;
        z = origin_.z + t * dir.z;
        if (x > -0.5 && x < 0.5 && z > -0.5 && z < 0.5 && t > 0 && t < dist - 1e-6)
            return false;

        t = (0.5 - origin_.y) / dir.y;
        x = origin_.x + t * dir.x;
        z = origin_.z + t * dir.z;
        if (x > -0.5 && x < 0.5 && z > -0.5 && z < 0.5 && t > 0 && t < dist - 1e-6)
            return false;

        t = (-0.5 - origin_.z) / dir.z;
        x = origin_.x + t * dir.x;
        y = origin_.y + t * dir.y;
        if (x > -0.5 && x < 0.5 && y > -0.5 && y < 0.5 && t > 0 && t < dist - 1e-6)
            return false;

        t = (0.5 - origin_.z) / dir.z;
        x = origin_.x + t * dir.x;
        y = origin_.y + t * dir.y;
        if (x > -0.5 && x < 0.5 && y > -0.5 && y < 0.5 && t > 0 && t < dist - 1e-6)
            return false;

        return true;
    }


    void MatchSyntheticShots(const detail::ImageFeatures &f1, const detail::ImageFeatures &f2,
                             vector<DMatch> &matches)
    {
        vector<DMatch> matches_;
        BruteForceMatcher<L2<int> > matcher;
        matcher.match(f1.descriptors, f2.descriptors, matches_);

        matches.clear();
        for (size_t i = 0; i < matches_.size(); ++i)
            if (f1.descriptors.at<int>(matches_[i].queryIdx) ==
                f2.descriptors.at<int>(matches_[i].trainIdx))
                matches.push_back(matches_[i]);
    }


    Mat CreateImage(const detail::ImageFeatures &features) {
        Mat img(features.img_size, CV_8U);
        img.setTo(0);

        for (size_t i = 0; i < features.keypoints.size(); ++i)
            circle(img, features.keypoints[i].pt, 1, Scalar::all(255), 1);

        return img;
    }


    namespace {

        void InitGlut() {
            static bool is_glut_initialized = false;
            if (!is_glut_initialized) {
                int argc = 0;
                glutInit(&argc, 0);
                is_glut_initialized = true;
            }
        }

        // The code was taken from http://www.gamedeception.net/threads/1876-Printing-Text-with-glut
        void PrintText(float x, float y, const char *text, void *font = GLUT_BITMAP_HELVETICA_12,
                       float r = 1.f, float g = 1.f, float b = 1.f, float a = 0.f)
        {
            if (!text || !strlen(text))
                return;

            bool blending = false;
            if (glIsEnabled(GL_BLEND))
                blending = true;

            glEnable(GL_BLEND);
            glColor4f(r, g, b, a);
            glRasterPos2f(x, y);

            while (*text) {
                glutBitmapCharacter(font, *text);
                text++;
            }

            if (!blending)
                glDisable(GL_BLEND);
        }

    } // namespace  


    void MonoViewer::Run() {
        if (!glfwInit())
            throw runtime_error("Can't initialize GLFW");

        if (!glfwOpenWindow(window_size_.width, window_size_.height,
                            0, 0, 0, 0, 0, 0, GLFW_WINDOW))
        {
            glfwTerminate();
            throw runtime_error("Can't create GLFW window");
        }

        glfwSetWindowTitle("Camera viewer");
        glfwSetKeyCallback(internal::MonoViewerKeyCallback);
        glfwSetWindowSizeCallback(internal::MonoViewerWindowSizeCallback);
        glfwSetMousePosCallback(internal::MonoViewerMousePosCallback);
        glfwSetMouseButtonCallback(internal::MonoViewerMouseButtonCallback);
        glfwEnable(GLFW_KEY_REPEAT);

        InitOpenGl();
        InitRun();

        detail::ImageFeatures features;

        while (is_running_ && glfwGetWindowParam(GLFW_OPENED)) {
            glClear(GL_COLOR_BUFFER_BIT);

            if (!scene_.empty()) {
                scene_->TakeShot(RigidCamera::FromLocalToWorld(K_, R_, T_), view_port_, features);
                glBegin(GL_POINTS);
                for (size_t i = 0; i < features.keypoints.size(); ++i) {
                    const Point2f &pt = features.keypoints[i].pt;
                    glVertex2f(pt.x, pt.y);
                }
                glEnd();
            }

            Mat_<double> rvec;
            Rodrigues(R_, rvec);
            stringstream text;
            text << "rvec = " << rvec;
            PrintText(10, 10, text.str().c_str());

            text.str("");
            text << "T = " << T_;
            PrintText(10, 70, text.str().c_str());

            glfwSwapBuffers();
        }

        glfwTerminate();
    }


    MonoViewer::MonoViewer() : cameras_(0), features_(0) {
        Mat_<double> K = Mat::eye(3, 3, CV_64F);
        K(0, 0) = 3000; K(0, 2) = 960;
        K(1, 1) = 3000; K(1, 2) = 540;
        set_K(K);

        set_R(Mat::eye(3, 3, CV_64F));
        set_T(Mat::zeros(3, 1, CV_64F));

        set_view_port(Rect(0, 0, 1920, 1080));
        set_window_size(Size(640, 360));

        set_move_speed(1e-1);
        set_rotation_speed(1e-2);

        InitGlut();
    }


    void MonoViewer::InitRun() {
        prev_key_ = -1;
        is_running_ = true;
        is_left_button_pressed_ = false;
    }


    void MonoViewer::InitOpenGl() {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        double aspect = (double)view_port_.width / view_port_.height;
        if (window_size_.width > aspect * window_size_.height)
            glViewport(0, 0, static_cast<GLsizei>(aspect * window_size_.height), static_cast<GLsizei>(window_size_.height));
        else
            glViewport(0, 0, static_cast<GLsizei>(window_size_.width), static_cast<GLsizei>(window_size_.width / aspect));

        glOrtho(view_port_.x, view_port_.br().x, view_port_.y, view_port_.br().y, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glDisable(GL_DEPTH_TEST);
    }


    void GLFWCALL internal::MonoViewerKeyCallback(int key, int state) {
        MonoViewer &v = the_mono_viewer();
        if (key == GLFW_KEY_ESC)
            v.is_running_ = false;
        else if (key == 'w' || key == 'W') {
            Mat_<double> oz(3, 1);
            oz(0, 0) = 0; oz(1, 0) = 0; oz(2, 0) = 1;
            oz = v.R_ * oz;
            v.T_ += oz * v.move_speed_;
        }
        else if (key == 's' || key == 'S') {
            Mat_<double> oz(3, 1);
            oz(0, 0) = 0; oz(1, 0) = 0; oz(2, 0) = 1;
            oz = v.R_ * oz;
            v.T_ -= oz * v.move_speed_;
        }
        else if (key == 'a' || key == 'A') {
            Mat_<double> ox(3, 1);
            ox(0, 0) = 1; ox(1, 0) = 0; ox(2, 0) = 0;
            ox = v.R_ * ox;
            v.T_ -= ox * v.move_speed_;
        }
        else if (key == 'd' || key == 'D') {
            Mat_<double> ox(3, 1);
            ox(0, 0) = 1; ox(1, 0) = 0; ox(2, 0) = 0;
            ox = v.R_ * ox;
            v.T_ += ox * v.move_speed_;
        }
        else if ((key == 't' || key == 'T') && key != v.prev_key_) {
            if (!v.scene_.empty()) {
                if (v.cameras_)
                    v.cameras_->push_back(RigidCamera::FromLocalToWorld(v.K_.clone(), v.R_.clone(), v.T_.clone()));
                if (v.features_) {
                    Ptr<detail::ImageFeatures> features = new detail::ImageFeatures();
                    v.scene_->TakeShot(RigidCamera::FromLocalToWorld(v.K_, v.R_, v.T_), v.view_port_, *features);
                    v.features_->insert(make_pair(v.features_->size(), features));
                }
                cout << "Took shot\n";
            }
            else
                cout << "Scene is empty\n";
        }
        else if ((key == 'h' || key == 'H') && key != v.prev_key_) {
            cout << "\nHot keys:\n"
                 << "    esc -- exit\n"
                 << "    w/s/a/d -- navigation\n"
                 << "    mouse_left_button -- orientation\n"
                 << "    t -- take a snapshot\n";
        }
        v.prev_key_ = key;
    }


    void GLFWCALL internal::MonoViewerWindowSizeCallback(int width, int height) {
        the_mono_viewer().window_size_.width = width;
        the_mono_viewer().window_size_.height = height;
        the_mono_viewer().InitOpenGl();
    }


    void GLFWCALL internal::MonoViewerMousePosCallback(int x, int y) {
        MonoViewer &v = the_mono_viewer();
        if (v.is_left_button_pressed_) {
            int dx = x - v.start_x_;
            int dy = y - v.start_y_;
            Mat_<double> ox(3, 1);
            ox(0, 0) = 1; ox(1, 0) = 0; ox(2, 0) = 0;
            ox *= CV_PI * dy / v.window_size_.width;
            Mat_<double> oy(3, 1);
            oy(0, 0) = 0; oy(1, 0) = 1; oy(2, 0) = 0;
            oy *= CV_PI * dx / v.window_size_.height;
            Mat R;
            Rodrigues(ox + oy, R);
            v.R_ = v.start_R_ * R;
        }
    }


    void GLFWCALL internal::MonoViewerMouseButtonCallback(int button, int state) {
        MonoViewer &v = the_mono_viewer();
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            if (state == GLFW_PRESS) {
                v.is_left_button_pressed_ = true;
                v.start_R_ = v.R_.clone();
                glfwGetMousePos(&v.start_x_, &v.start_y_);
            }
            else {
                v.is_left_button_pressed_ = false;
            }
        }
    }


    MonoViewer& the_mono_viewer() {
        static MonoViewer instance;
        return instance;
    }  


    void StereoViewer::Run() {
        if (!glfwInit())
            throw runtime_error("Can't initialize GLFW");

        if (!glfwOpenWindow(window_size_.width, window_size_.height,
                            0, 0, 0, 0, 0, 0, GLFW_WINDOW))
        {
            glfwTerminate();
            throw runtime_error("Can't create GLFW window");
        }

        glfwSetWindowTitle("Stereo camera viewer");
        glfwSetKeyCallback(internal::StereoViewerKeyCallback);
        glfwSetWindowSizeCallback(internal::StereoViewerWindowSizeCallback);
        glfwSetMousePosCallback(internal::StereoViewerMousePosCallback);
        glfwSetMouseButtonCallback(internal::StereoViewerMouseButtonCallback);
        glfwSetMouseWheelCallback(internal::StereoViewerMouseWheelCallback);
        glfwEnable(GLFW_KEY_REPEAT);

        InitOpenGl();
        InitRun();

        detail::ImageFeatures left_features, right_features;

        while (is_running_ && glfwGetWindowParam(GLFW_OPENED)) {
            glClear(GL_COLOR_BUFFER_BIT);

            if (!scene_.empty()) {
                RigidCamera left_camera(K_, Rg_.t(), -Rg_.t() * Tg_);
                RigidCamera right_camera(K_, R_.t() * Rg_.t(), -R_.t() * Rg_.t() * Tg_ - R_.t() * T_);

                scene_->TakeShot(left_camera, view_port_, left_features);
                scene_->TakeShot(right_camera, view_port_, right_features);

                glBegin(GL_POINTS);
                for (size_t i = 0; i < left_features.keypoints.size(); ++i) {
                    const Point2f &pt = left_features.keypoints[i].pt;
                    glVertex2f(pt.x, pt.y);
                }
                for (size_t i = 0; i < right_features.keypoints.size(); ++i) {
                    const Point2f &pt = right_features.keypoints[i].pt;
                    glVertex2f(pt.x + view_port_.width, pt.y);
                }
                glEnd();
            }

            Mat_<double> rvec;
            Rodrigues(Rg_, rvec);
            stringstream text;
            text << "rvec_g = " << rvec;
            PrintText(10.f, 10.f, text.str().c_str());

            text.str("");
            text << "Tg = " << Tg_;
            PrintText(10.f, 70.f, text.str().c_str());

            Rodrigues(R_, rvec);
            text.str("");
            text << "rvec = " << rvec;
            PrintText(static_cast<float>(10 + view_port_.width), 10.f, text.str().c_str());

            text.str("");
            text << "T = " << T_;
            PrintText(static_cast<float>(10 + view_port_.width), 70.f, text.str().c_str());

            glfwSwapBuffers();
        }

        glfwTerminate();
    }


    StereoViewer::StereoViewer() : left_cameras_(0), right_cameras_(0),
                                   left_features_(0), right_features_(0)
    {
        Mat_<double> K = Mat::eye(3, 3, CV_64F);
        K(0, 0) = 3000; K(0, 2) = 960;
        K(1, 1) = 3000; K(1, 2) = 540;
        set_K(K);

        set_R(Mat::eye(3, 3, CV_64F));

        Mat_<double> T(3, 1);
        T(0, 0) = -1; T(1, 0) = 0; T(2, 0) = 0;
        set_T(T);

        set_Rg(Mat::eye(3, 3, CV_64F));
        set_Tg(Mat::zeros(3, 1, CV_64F));

        set_view_port(Rect(0, 0, 1920, 1080));
        set_window_size(Size(640 * 2, 360));

        set_move_speed(1e-1);
        set_rotation_speed(1e-2);
        set_baseline_speed(1e-2);

        InitGlut();
    }


    void StereoViewer::InitRun() {
        prev_wheel_pos_ = 0;
        prev_key_ = -1;
        orientation_mode_ = ORIENTATION_MODE_STEREO_PAIR;
        is_running_ = true;
        is_left_button_pressed_ = false;
    }


    void StereoViewer::InitOpenGl() {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        double aspect = 2 * (double)view_port_.width / view_port_.height;
        if (window_size_.width > aspect * window_size_.height)
            glViewport(0, 0, static_cast<GLsizei>(aspect * window_size_.height), static_cast<GLsizei>(window_size_.height));
        else
            glViewport(0, 0, static_cast<GLsizei>(window_size_.width), static_cast<GLsizei>(window_size_.width / aspect));

        glOrtho(view_port_.x, view_port_.br().x + view_port_.width,
                view_port_.y, view_port_.br().y, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glDisable(GL_DEPTH_TEST);
    }


    void GLFWCALL internal::StereoViewerKeyCallback(int key, int state) {
        StereoViewer &v = the_stereo_viewer();
        if (key == GLFW_KEY_ESC)
            v.is_running_ = false;
        else if (key == 'w' || key == 'W') {
            Mat_<double> oz(3, 1);
            oz(0, 0) = 0; oz(1, 0) = 0; oz(2, 0) = 1;
            oz = v.Rg_ * oz;
            if (v.orientation_mode_ == StereoViewer::ORIENTATION_MODE_STEREO_PAIR)
                v.Tg_ += oz * v.move_speed_;
            else
                v.T_ += oz * v.move_speed_;
        }
        else if (key == 's' || key == 'S') {
            Mat_<double> oz(3, 1);
            oz(0, 0) = 0; oz(1, 0) = 0; oz(2, 0) = 1;
            oz = v.Rg_ * oz;
            if (v.orientation_mode_ == StereoViewer::ORIENTATION_MODE_STEREO_PAIR)
                v.Tg_ -= oz * v.move_speed_;
            else
                v.T_ -= oz * v.move_speed_;
        }
        else if (key == 'a' || key == 'A') {
            Mat_<double> ox(3, 1);
            ox(0, 0) = 1; ox(1, 0) = 0; ox(2, 0) = 0;
            ox = v.Rg_ * ox;
            if (v.orientation_mode_ == StereoViewer::ORIENTATION_MODE_STEREO_PAIR)
                v.Tg_ -= ox * v.move_speed_;
            else
                v.T_ -= ox * v.move_speed_;
        }
        else if (key == 'd' || key == 'D') {
            Mat_<double> ox(3, 1);
            ox(0, 0) = 1; ox(1, 0) = 0; ox(2, 0) = 0;
            ox = v.Rg_ * ox;
            if (v.orientation_mode_ == StereoViewer::ORIENTATION_MODE_STEREO_PAIR)
                v.Tg_ += ox * v.move_speed_;
            else
                v.T_ += ox * v.move_speed_;
        }
        else if (key == GLFW_KEY_UP) {
            Mat_<double> oy(3, 1);
            oy(0, 0) = 0; oy(1, 0) = 1; oy(2, 0) = 0;
            oy = v.Rg_ * oy;
            if (v.orientation_mode_ == StereoViewer::ORIENTATION_MODE_STEREO_PAIR)
                v.Tg_ += oy * v.move_speed_;
            else
                v.T_ += oy * v.move_speed_;
        }
        else if (key == GLFW_KEY_DOWN) {
            Mat_<double> oy(3, 1);
            oy(0, 0) = 0; oy(1, 0) = 1; oy(2, 0) = 0;
            oy = v.Rg_ * oy;
            if (v.orientation_mode_ == StereoViewer::ORIENTATION_MODE_STEREO_PAIR)
                v.Tg_ -= oy * v.move_speed_;
            else
                v.T_ -= oy * v.move_speed_;
        }
        else if (key == GLFW_KEY_LEFT) {
            Mat_<double> oz(3, 1);
            oz(0, 0) = 0; oz(1, 0) = 0; oz(2, 0) = 1;
            oz *= v.rotation_speed_;

            Mat R;
            Rodrigues(oz, R);

            if (v.orientation_mode_ == StereoViewer::ORIENTATION_MODE_STEREO_PAIR)
                v.Rg_ *= R;
            else if (v.orientation_mode_ == StereoViewer::ORIENTATION_MODE_RELATIVE)
                v.R_ *= R;
        }
        else if (key == GLFW_KEY_RIGHT) {
            Mat_<double> oz(3, 1);
            oz(0, 0) = 0; oz(1, 0) = 0; oz(2, 0) = 1;
            oz *= -v.rotation_speed_;

            Mat R;
            Rodrigues(oz, R);

            if (v.orientation_mode_ == StereoViewer::ORIENTATION_MODE_STEREO_PAIR)
                v.Rg_ *= R;
            else if (v.orientation_mode_ == StereoViewer::ORIENTATION_MODE_RELATIVE)
                v.R_ *= R;
        }
        else if ((key == 't' || key == 'T') && key != v.prev_key_) {
            if (!v.scene_.empty()) {
                RigidCamera left_camera(v.K_, v.Rg_.t(), -v.Rg_.t() * v.Tg_);
                RigidCamera right_camera(v.K_.clone(), v.R_.t() * v.Rg_.t(), -v.R_.t() * v.Rg_.t() * v.Tg_ - v.R_.t() * v.T_);
                if (v.left_cameras_)
                    v.left_cameras_->push_back(left_camera);
                if (v.right_cameras_)
                    v.right_cameras_->push_back(right_camera);
                if (v.left_features_) {
                    Ptr<detail::ImageFeatures> features = new detail::ImageFeatures();
                    v.scene_->TakeShot(left_camera, v.view_port_, *features);
                    v.left_features_->insert(make_pair(v.left_features_->size(), features));
                }
                if (v.right_features_) {
                    Ptr<detail::ImageFeatures> features = new detail::ImageFeatures();
                    v.scene_->TakeShot(right_camera, v.view_port_, *features);
                    v.right_features_->insert(make_pair(v.right_features_->size(), features));
                }
                cout << "Took shot\n";
            }
            else
                cout << "Scene is empty\n";
        }
        else if (key == GLFW_KEY_F1 && key != v.prev_key_) {
            v.orientation_mode_ = StereoViewer::ORIENTATION_MODE_STEREO_PAIR;
            cout << "Stereo pair orientation mode\n";
        }
        else if (key == GLFW_KEY_F2 && key != v.prev_key_) {
            v.orientation_mode_ = StereoViewer::ORIENTATION_MODE_RELATIVE;
            cout << "Relative orientation mode\n";
        }
        else if ((key == 'h' || key == 'H') && key != v.prev_key_) {
            cout << "\nHot keys:\n"
                 << "    esc -- exit\n"
                 << "    w/s/a/d -- navigation (OX, OZ)\n"
                 << "    mouse_left_button -- orientation (OX, OY)\n"
                 << "    up/down -- ext. navigation (OY)\n"
                 << "    left/right -- ext. orientation (OZ)\n"
                 << "    mouse_wheel -- baseline\n"
                 << "    F1 -- turn on stereo pair orientation mode (default)\n"
                 << "    F2 -- turn on relative orientation mode\n"
                 << "    t -- take a snapshot\n";
        }

        v.prev_key_ = key;
    }


    void GLFWCALL internal::StereoViewerWindowSizeCallback(int width, int height) {
        the_stereo_viewer().window_size_.width = width;
        the_stereo_viewer().window_size_.height = height;
        the_stereo_viewer().InitOpenGl();
    }


    void GLFWCALL internal::StereoViewerMousePosCallback(int x, int y) {
        StereoViewer &v = the_stereo_viewer();
        if (v.is_left_button_pressed_) {
            int dx = x - v.start_x_;
            int dy = y - v.start_y_;
            Mat_<double> ox(3, 1);
            ox(0, 0) = 1; ox(1, 0) = 0; ox(2, 0) = 0;
            ox *= CV_PI * dy / v.window_size_.width;
            Mat_<double> oy(3, 1);
            oy(0, 0) = 0; oy(1, 0) = 1; oy(2, 0) = 0;
            oy *= CV_PI * dx / v.window_size_.height;
            Mat R;
            Rodrigues(ox + oy, R);

            if (v.orientation_mode_ == StereoViewer::ORIENTATION_MODE_STEREO_PAIR)
                v.Rg_ = v.start_orientation_ * R;
            else if (v.orientation_mode_ == StereoViewer::ORIENTATION_MODE_RELATIVE)
                v.R_ = v.start_orientation_ * R;
        }
    }


    void GLFWCALL internal::StereoViewerMouseButtonCallback(int button, int state) {
        StereoViewer &v = the_stereo_viewer();
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            if (state == GLFW_PRESS) {
                v.is_left_button_pressed_ = true;

                if (v.orientation_mode_ == StereoViewer::ORIENTATION_MODE_STEREO_PAIR)
                    v.start_orientation_ = v.Rg_.clone();
                else if (v.orientation_mode_ == StereoViewer::ORIENTATION_MODE_RELATIVE)
                    v.start_orientation_ = v.R_.clone();

                glfwGetMousePos(&v.start_x_, &v.start_y_);
            }
            else {
                v.is_left_button_pressed_ = false;
            }
        }
    }


    void GLFWCALL internal::StereoViewerMouseWheelCallback(int pos) {
        StereoViewer &v = the_stereo_viewer();
        v.T_ *= 1 + (pos - v.prev_wheel_pos_) * v.baseline_speed_;
        v.prev_wheel_pos_ = pos;
    }


    StereoViewer& the_stereo_viewer() {
        static StereoViewer instance;
        return instance;
    }


    void KeypointsExtractor::Run() {
        if (!glfwInit())
            throw runtime_error("Can't initialize GLFW");

        if (!glfwOpenWindow(window_size_.width, window_size_.height,
                            0, 0, 0, 0, 0, 0, GLFW_WINDOW))
        {
            glfwTerminate();
            throw runtime_error("Can't create GLFW window");
        }

        glfwSetWindowTitle("Keypoints extractor");
        glfwSetKeyCallback(internal::KeypointsExtractorKeyCallback);
        glfwSetWindowSizeCallback(internal::KeypointsExtractorWindowSizeCallback);
        glfwSetMouseButtonCallback(internal::KeypointsExtractorMouseButtonCallback);

        InitOpenGl();
        InitRun();

        while (is_running_ && glfwGetWindowParam(GLFW_OPENED)) {
            glClear(GL_COLOR_BUFFER_BIT);
            glColor3f(1.f, 1.f, 1.f);
            glBegin(GL_QUADS);
            glTexCoord2f(0.f, 0.f); glVertex2f(0.f, (float)image_.rows);
            glTexCoord2f(1.f, 0.f); glVertex2f((float)image_.cols, (float)image_.rows);
            glTexCoord2f(1.f, 1.f); glVertex2f((float)image_.cols, 0.f);
            glTexCoord2f(0.f, 1.f); glVertex2f(0.f, 0.f);
            glEnd();
            if (keypoints_) {
                glColor3f(1.f, 0.f, 0.f);
                glPointSize(5.f);
                glBegin(GL_POINTS);
                for (size_t i = 0; i < keypoints_->size(); ++i) {
                    const Point2f &pt = (*keypoints_)[i];
                    glVertex2f(pt.x, image_.rows - pt.y);
                }
                glEnd();
            }
            glfwSwapBuffers();
        }

        glfwTerminate();
    }


    KeypointsExtractor::KeypointsExtractor() : keypoints_(0) {
        set_window_size(Size(640, 480));
    }


    void KeypointsExtractor::InitOpenGl() {
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &texture_);
        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, 3, image_.cols, image_.rows, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, image_.data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, texture_);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glViewport(0, 0, window_size_.width, window_size_.height);
        glOrtho(0, image_.cols, 0, image_.rows, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glDisable(GL_DEPTH_TEST);
    }


    void KeypointsExtractor::InitRun() {
        is_running_ = true;
    }


    void GLFWCALL internal::KeypointsExtractorKeyCallback(int key, int state) {
        KeypointsExtractor &ke = the_keypoints_extractor();
        if (key == GLFW_KEY_ESC)
            ke.is_running_ = false;
    }


    void GLFWCALL internal::KeypointsExtractorWindowSizeCallback(int width, int height) {
        the_keypoints_extractor().window_size_.width = width;
        the_keypoints_extractor().window_size_.height = height;
        the_keypoints_extractor().InitOpenGl();
    }


    void GLFWCALL internal::KeypointsExtractorMouseButtonCallback(int button, int state) {
        KeypointsExtractor &ke = the_keypoints_extractor();
        if (button == GLFW_MOUSE_BUTTON_LEFT && state == GLFW_PRESS) {
            if (ke.keypoints_) {
                int x, y;
                glfwGetMousePos(&x, &y);
                ke.keypoints_->push_back(Point2f((float)x / ke.window_size_.width * ke.image_.cols,
                                                 (float)y / ke.window_size_.height * ke.image_.rows));
            }
        }
    }


    KeypointsExtractor& the_keypoints_extractor() {
        static KeypointsExtractor instance;
        return instance;
    }


    void FeaturesMatcher::Run() {
        if (!glfwInit())
            throw runtime_error("Can't initialize GLFW");

        if (!glfwOpenWindow(window_size_.width, window_size_.height,
                            0, 0, 0, 0, 0, 0, GLFW_WINDOW))
        {
            glfwTerminate();
            throw runtime_error("Can't create GLFW window");
        }

        glfwSetWindowTitle("Features matcher");
        glfwSetKeyCallback(internal::FeaturesMatcherKeyCallback);
        glfwSetWindowSizeCallback(internal::FeaturesMatcherWindowSizeCallback);
        glfwSetMouseButtonCallback(internal::FeaturesMatcherMouseButtonCallback);

        InitOpenGl();
        InitRun();

        while (is_running_ && glfwGetWindowParam(GLFW_OPENED)) {
            glClear(GL_COLOR_BUFFER_BIT);

            glColor3f(1.f, 1.f, 1.f);

            glBindTexture(GL_TEXTURE_2D, texture1_);
            glBegin(GL_QUADS);
            glTexCoord2f(0.f, 0.f); glVertex2f(0.f, (float)image1_.rows);
            glTexCoord2f(1.f, 0.f); glVertex2f((float)image1_.cols, (float)image1_.rows);
            glTexCoord2f(1.f, 1.f); glVertex2f((float)image1_.cols, 0.f);
            glTexCoord2f(0.f, 1.f); glVertex2f(0.f, 0.f);
            glEnd();

            glBindTexture(GL_TEXTURE_2D, texture2_);
            glBegin(GL_QUADS);
            glTexCoord2f(0.f, 0.f); glVertex2f((float)image1_.cols, (float)image2_.rows);
            glTexCoord2f(1.f, 0.f); glVertex2f((float)(image1_.cols + image2_.cols), (float)image2_.rows);
            glTexCoord2f(1.f, 1.f); glVertex2f((float)(image1_.cols + image2_.cols), 0.f);
            glTexCoord2f(0.f, 1.f); glVertex2f((float)image1_.cols, 0.f);
            glEnd();

            if (keypoints1_) {
                glColor3f(1.f, 0.f, 0.f);
                glPointSize(5.f);
                glBegin(GL_POINTS);
                for (size_t i = 0; i < keypoints1_->size(); ++i) {
                    const Point2f &pt = (*keypoints1_)[i];
                    glVertex2f(pt.x, std::max(image1_.rows, image2_.rows) - pt.y);
                }
                glEnd();
            }

            if (keypoints2_) {
                glColor3f(1.f, 0.f, 0.f);
                glPointSize(5.f);
                glBegin(GL_POINTS);
                for (size_t i = 0; i < keypoints2_->size(); ++i) {
                    const Point2f &pt = (*keypoints2_)[i];
                    glVertex2f(image1_.cols + pt.x, std::max(image1_.rows, image2_.rows) - pt.y);
                }
                glEnd();
            }

            if (matches_ && keypoints1_ && keypoints2_) {
                glColor3f(1.f, 0.f, 0.f);
                glBegin(GL_LINES);
                for (size_t i = 0; i < matches_->size(); ++i) {
                    const Point2f &pt1 = (*keypoints1_)[(*matches_)[i].queryIdx];
                    const Point2f &pt2 = (*keypoints2_)[(*matches_)[i].trainIdx];
                    glVertex2f(pt1.x, std::max(image1_.rows, image2_.rows) - pt1.y);
                    glVertex2f(image1_.cols + pt2.x, std::max(image1_.rows, image2_.rows) - pt2.y);
                }
                glEnd();
            }

            if (is_keypoint1_selected_) {
                glColor3f(1.f, 0.f, 0.f);
                glBegin(GL_LINES);
                const Point2f &pt = (*keypoints1_)[prev_keypoint_index_];
                glVertex2f(pt.x, std::max(image1_.rows, image2_.rows) - pt.y);
                int x, y; glfwGetMousePos(&x, &y);
                float u, v; ScreenToLocal(x, y, u, v);
                glVertex2f(u, std::max(image1_.rows, image2_.rows) - v);
                glEnd();
            }

            if (is_keypoint2_selected_) {
                glColor3f(1.f, 0.f, 0.f);
                glBegin(GL_LINES);
                const Point2f &pt = (*keypoints2_)[prev_keypoint_index_];
                glVertex2f(pt.x + image1_.cols, std::max(image1_.rows, image2_.rows) - pt.y);
                int x, y; glfwGetMousePos(&x, &y);
                float u, v; ScreenToLocal(x, y, u, v);
                glVertex2f(u, std::max(image1_.rows, image2_.rows) - v);
                glEnd();
            }

            glfwSwapBuffers();
        }

        glfwTerminate();
    }


    FeaturesMatcher::FeaturesMatcher() : matches_(0) {
        set_window_size(Size(1280, 480));
    }


    void FeaturesMatcher::InitOpenGl() {
        glEnable(GL_TEXTURE_2D);

        glGenTextures(1, &texture1_);
        glBindTexture(GL_TEXTURE_2D, texture1_);
        glTexImage2D(GL_TEXTURE_2D, 0, 3, image1_.cols, image1_.rows, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, image1_.data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glGenTextures(1, &texture2_);
        glBindTexture(GL_TEXTURE_2D, texture2_);
        glTexImage2D(GL_TEXTURE_2D, 0, 3, image2_.cols, image2_.rows, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, image2_.data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glViewport(0, 0, window_size_.width, window_size_.height);
        glOrtho(0, image1_.cols + image2_.cols, 0, std::max(image1_.rows, image2_.rows), -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glDisable(GL_DEPTH_TEST);
    }


    void FeaturesMatcher::InitRun() {
        is_running_ = true;
        is_keypoint1_selected_ = false;
        is_keypoint2_selected_ = false;
    }


    void FeaturesMatcher::ScreenToLocal(int x, int y, float &u, float &v) {
        u = (float)x / window_size_.width * (image1_.cols + image2_.cols);
        v = (float)y / window_size_.height * std::max(image1_.rows, image2_.rows);
    }


    void GLFWCALL internal::FeaturesMatcherKeyCallback(int key, int state) {
        FeaturesMatcher &fm = the_features_matcher();
        if (key == GLFW_KEY_ESC) {
            if (fm.is_keypoint1_selected_)
                fm.is_keypoint1_selected_ = false;
            else if (fm.is_keypoint2_selected_)
                fm.is_keypoint2_selected_ = false;
            else
                fm.is_running_ = false;
        }
    }


    void GLFWCALL internal::FeaturesMatcherWindowSizeCallback(int width, int height) {
        the_features_matcher().window_size_.width = width;
        the_features_matcher().window_size_.height = height;
        the_features_matcher().InitOpenGl();
    }


    void GLFWCALL internal::FeaturesMatcherMouseButtonCallback(int button, int state) {
        FeaturesMatcher &fm = the_features_matcher();
        if (button == GLFW_MOUSE_BUTTON_LEFT && state == GLFW_PRESS) {
            if (fm.matches_) {
                int x, y; glfwGetMousePos(&x, &y);
                float u, v; fm.ScreenToLocal(x, y, u, v);

                int keypoint_index = -1;
                float min_dist = numeric_limits<float>::max();

                if (u < fm.image1_.cols && fm.keypoints1_) {
                    for (size_t i = 0; i < fm.keypoints1_->size(); ++i) {
                        const Point2f &pt = (*fm.keypoints1_)[i];
                        float dist = sqr(u - pt.x) + sqr(v - pt.y);
                        if (dist < min_dist) {
                            keypoint_index = i;
                            min_dist = dist;
                        }
                    }
                    if (keypoint_index != -1) {
                        if (fm.is_keypoint2_selected_) {
                            fm.matches_->push_back(DMatch(keypoint_index, fm.prev_keypoint_index_, 0));
                            fm.is_keypoint2_selected_ = false;
                        }
                        else {
                            fm.is_keypoint1_selected_ = true;
                            fm.prev_keypoint_index_ = keypoint_index;
                        }
                    }
                }
                else if (u >= fm.image1_.cols && fm.keypoints2_) {
                    u -= (float)fm.image1_.cols;
                    for (size_t i = 0; i < fm.keypoints2_->size(); ++i) {
                        const Point2f &pt = (*fm.keypoints2_)[i];
                        float dist = sqr(u - pt.x) + sqr(v - pt.y);
                        if (dist < min_dist) {
                            keypoint_index = i;
                            min_dist = dist;
                        }
                    }
                    if (keypoint_index != -1) {
                        if (fm.is_keypoint1_selected_) {
                            fm.matches_->push_back(DMatch(fm.prev_keypoint_index_, keypoint_index, 0));
                            fm.is_keypoint1_selected_ = false;
                        }
                        else {
                            fm.is_keypoint2_selected_ = true;
                            fm.prev_keypoint_index_ = keypoint_index;
                        }
                    }
                }
            }
        }
    }


    FeaturesMatcher& the_features_matcher() {
        static FeaturesMatcher instance;
        return instance;
    }

} // namespace evaluation
} // namespace autocalib
