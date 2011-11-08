#include "precomp.h"
#include <include/core.h>

using namespace std;
using namespace cv;

namespace autocalib {

    RigidCamera RigidCamera::FromProjectiveMat(const Mat &P) {
        CV_Assert(P.size() == Size(4, 3) && P.type() == CV_64F);

        Mat_<double> K, R, T;

        RQDecomp3x3(P(Rect(0, 0, 3, 3)), K, R);
        T = K.inv() * P.col(3);
        K /= K(2, 2);

        if (K(0, 0) < 0 && K(1, 1) < 0) {
            K.col(0) *= -1; K.col(1) *= -1;
            R.row(0) *= -1; R.row(1) *= -1;
            T(0, 0) *= -1;
            T(1, 0) *= -1;
        }

        return RigidCamera(K, R, T);
    }


    Mat CalibRotationalCameraLinear(const HomographiesP2 &Hs, double *residual_error) {
        int num_Hs = (int)Hs.size();
        if (num_Hs < 1)
            throw runtime_error("Need at least one homography");

        // Normalize homographies
        vector<Mat> Hs_normed;
        for (HomographiesP2::const_iterator iter = Hs.begin(); iter != Hs.end(); ++iter) {
            Mat H = iter->second;
            CV_Assert(H.size() == Size(3, 3) && H.type() == CV_64F);

            double det = determinant(H);
            double norm = pow(abs(det), 1. / 3.) * (det < 0. ? -1. : 1.);
            Hs_normed.push_back(H / norm);
        }

        Mat_<double> A(6 * num_Hs, 5);
        Mat_<double> b(6 * num_Hs, 1);
        b.setTo(0);

        static const int lut[][3] = {{0, 1, 2}, {-1, 3, 4}, {-1, -1, -1}};

        int eq_idx = 0;
        for (int H_idx = 0; H_idx < num_Hs; ++H_idx) {
            Mat_<double> H = Hs_normed[H_idx];
            for (int r1 = 0; r1 < 3; ++r1) {
                for (int r2 = r1; r2 < 3; ++r2) {
                    A(eq_idx, 0) = H(r1, 0) * H(r2, 0);
                    A(eq_idx, 1) = H(r1, 0) * H(r2, 1) + H(r1, 1) * H(r2, 0);
                    A(eq_idx, 2) = H(r1, 0) * H(r2, 2) + H(r1, 2) * H(r2, 0);
                    A(eq_idx, 3) = H(r1, 1) * H(r2, 1);
                    A(eq_idx, 4) = H(r1, 1) * H(r2, 2) + H(r1, 2) * H(r2, 1);

                    if (r1 == 2 && r2 == 2)
                        b(eq_idx, 0) = 1 - H(r1, 2) * H(r2, 2);
                    else {
                        A(eq_idx, lut[r1][r2]) -= 1;
                        b(eq_idx, 0) = -H(r1, 2) * H(r2, 2);
                    }

                    b(eq_idx, 0) /= norm(A.row(eq_idx)) + 1;
                    A.row(eq_idx) /= norm(A.row(eq_idx)) + 1;

                    eq_idx++;
                }
            }
        }

        Mat_<double> x;
        solve(A, b, x, DECOMP_SVD);
        Mat err = A * x - b;

        double residual_error_ = sqrt(err.dot(err) / b.dot(b));
        if (residual_error)
            *residual_error = residual_error_;
        AUTOCALIB_LOG(cout << "solve() norm(A*x - b) / norm(b) = " << residual_error_ << endl);

        // Dual Image of the Absolute Conic == K * K.t()
        Mat_<double> diac = Mat::eye(3, 3, CV_64F);
        diac(0, 0) = x(0, 0);
        diac(0, 1) = diac(1, 0) = x(1, 0);
        diac(0, 2) = diac(2, 0) = x(2, 0);
        diac(1, 1) = x(3, 0);
        diac(1, 2) = diac(2, 1) = x(4, 0);

        AUTOCALIB_LOG(Mat evals; Mat evecs;
            eigen(diac, evals, evecs);
            cout << "DIAC = K * K.t() = \n" << diac << endl;
            cout << "DIAC evecs = \n" << evecs << endl;
            cout << "DIAC evals = \n" << evals << endl);

        Mat K = DecomposeUUt(diac);
        if (K.empty())
            throw runtime_error("DIAC isn't positive definite");
        return K;
    }


    Mat CalibRotationalCameraLinearNoSkew(const HomographiesP2 &Hs, double *residual_error) {
        int num_Hs = (int)Hs.size();
        cout << num_Hs << endl;
        if (num_Hs < 1)
            throw runtime_error("Need at least one homography");

        // Normalize and transpose homographies
        vector<Mat> Hs_normed_t;
        for (HomographiesP2::const_iterator iter = Hs.begin(); iter != Hs.end(); ++iter) {
            Mat H = iter->second;
            CV_Assert(H.size() == Size(3, 3) && H.type() == CV_64F);

            double det = determinant(H);
            double norm = pow(abs(det), 1. / 3.) * (det < 0. ? -1. : 1.);
            Hs_normed_t.push_back((H / norm).t());
        }

        Mat_<double> A(6 * num_Hs, 4);
        Mat_<double> b(6 * num_Hs, 1);
        A.setTo(0);
        b.setTo(0);

        static const int lut[][3] = {{0, -1, 1}, {-1, 2, 3}, {-1, -1, -1}};

        int eq_idx = 0;
        for (int H_idx = 0; H_idx < num_Hs; ++H_idx) {
            Mat_<double> Ht = Hs_normed_t[H_idx];

            for (int r1 = 0; r1 < 3; ++r1) {
                for (int r2 = r1; r2 < 3; ++r2) {
                    A(eq_idx, 0) = Ht(r1, 0) * Ht(r2, 0);
                    A(eq_idx, 1) = Ht(r1, 0) * Ht(r2, 2) + Ht(r1, 2) * Ht(r2, 0);
                    A(eq_idx, 2) = Ht(r1, 1) * Ht(r2, 1);
                    A(eq_idx, 3) = Ht(r1, 1) * Ht(r2, 2) + Ht(r1, 2) * Ht(r2, 1);

                    if (r1 == 2 && r2 == 2)
                        b(eq_idx, 0) = 1 - Ht(r1, 2) * Ht(r2, 2);
                    else if (r1 == 0 && r2 == 1)
                        b(eq_idx, 0) = -Ht(r1, 2) * Ht(r2, 2);
                    else {
                        A(eq_idx, lut[r1][r2]) -= 1;
                        b(eq_idx, 0) = -Ht(r1, 2) * Ht(r2, 2);
                    }

                    b(eq_idx, 0) /= norm(A.row(eq_idx)) + 1;
                    A.row(eq_idx) /= norm(A.row(eq_idx)) + 1;

                    eq_idx++;
                }
            }
        }

        Mat_<double> x;
        solve(A, b, x, DECOMP_SVD);
        Mat err = A * x - b;

        double residual_error_ = sqrt(err.dot(err) / b.dot(b));
        if (residual_error)
            *residual_error = residual_error_;
        AUTOCALIB_LOG(cout << "solve() norm(A*x - b) / norm(b) = " << residual_error_ << endl);

        // Image of the Absolute Conic == (K * K.t()).inv()
        Mat_<double> iac = Mat::eye(3, 3, CV_64F);
        iac(0, 0) = x(0, 0);
        iac(0, 2) = iac(2, 0) = x(1, 0);
        iac(1, 1) = x(2, 0);
        iac(1, 2) = iac(2, 1) = x(3, 0);

        AUTOCALIB_LOG(
            Mat evals; Mat evecs;
            eigen(iac, evals, evecs);
            cout << "IAC = (K * K.t()).inv() =\n" << iac << endl;
            cout << "IAC evecs = \n" << evecs << endl;
            cout << "IAC evals = \n" << evals << endl);

        Mat K_inv_t = DecomposeCholesky(iac);
        if (K_inv_t.empty())
            throw runtime_error("IAC isn't positive definite");

        Mat_<double> K = K_inv_t.inv().t();
        K /= K(2, 2);        

        return K;
    }


    namespace {

        class ReprojError_FixedK_OnlyR {
        public:
            ReprojError_FixedK_OnlyR(const FeaturesCollection &features,
                                     const MatchesCollection &matches,
                                     int params_to_refine,
                                     const vector<int> &Rs_indices)
                    : features_(&features), matches_(&matches), params_to_refine_(params_to_refine),
                      step_(1e-4)
            {
                num_matches_ = 0;
                for (MatchesCollection::const_iterator view = matches_->begin();
                     view != matches_->end(); ++view)
                    num_matches_ += (int)view->second->size();

                Rs_indices_inv_.assign(*max_element(Rs_indices.begin(), Rs_indices.end()) + 1, -1);
                for (size_t i = 0; i < Rs_indices.size(); ++i)
                    Rs_indices_inv_[Rs_indices[i]] = i;
            }

            void operator()(const Mat &arg, Mat &err);
            void Jacobian(const Mat &arg, Mat &jac);

            int dimension() const { return num_matches_ * 2; }

        private:
            const FeaturesCollection *features_;
            const MatchesCollection *matches_;
            int num_matches_;
            int params_to_refine_;
            vector<int> Rs_indices_inv_;

            const double step_;
            Mat_<double> err_;
        };


        void ReprojError_FixedK_OnlyR::operator()(const Mat &arg, Mat &err) {
            Mat_<double> arg_(arg);

            err.create(dimension(), 1, CV_64F);
            Mat_<double> err_(err);

            Mat_<double> K = Mat::eye(3, 3, CV_64F);
            K(0, 0) = arg_(0, 0);
            K(0, 1) = arg_(0, 1);
            K(0, 2) = arg_(0, 2);
            K(1, 1) = arg_(0, 3);
            K(1, 2) = arg_(0, 4);
            Mat K_inv = K.inv();

            int pos = 0;
            for (MatchesCollection::const_iterator view = matches_->begin();
                 view != matches_->end(); ++view)
            {
                int img_from = view->first.first;
                const vector<KeyPoint> &kps_from = features_->find(img_from)->second->keypoints;
                Mat_<double> rvec_from(1, 3);
                if (Rs_indices_inv_[img_from] > 0) {
                    rvec_from(0, 0) = arg_(0, 5 + 3 * (Rs_indices_inv_[img_from] - 1));
                    rvec_from(0, 1) = arg_(0, 5 + 3 * (Rs_indices_inv_[img_from] - 1) + 1);
                    rvec_from(0, 2) = arg_(0, 5 + 3 * (Rs_indices_inv_[img_from] - 1) + 2);
                }
                else
                    rvec_from.setTo(0);
                Mat R_from;
                Rodrigues(rvec_from, R_from);

                int img_to = view->first.second;
                const vector<KeyPoint> &kps_to = features_->find(img_to)->second->keypoints;
                Mat_<double> rvec_to(1, 3);
                if (Rs_indices_inv_[img_to] > 0) {
                    rvec_to(0, 0) = arg_(0, 5 + 3 * (Rs_indices_inv_[img_to] - 1));
                    rvec_to(0, 1) = arg_(0, 5 + 3 * (Rs_indices_inv_[img_to] - 1) + 1);
                    rvec_to(0, 2) = arg_(0, 5 + 3 * (Rs_indices_inv_[img_to] - 1) + 2);
                }
                else
                    rvec_to.setTo(0);
                Mat R_to;
                Rodrigues(rvec_to, R_to);

                Mat_<double> M = K * R_from * R_to.t() * K_inv;

                const vector<DMatch> &matches = *(view->second);
                for (size_t i = 0; i < matches.size(); ++i, ++pos) {
                    const Point2f &p1 = kps_from[matches[i].queryIdx].pt;
                    const Point2f &p2 = kps_to[matches[i].trainIdx].pt;
                    double x = M(0, 0) * p2.x + M(0, 1) * p2.y + M(0, 2);
                    double y = M(1, 0) * p2.x + M(1, 1) * p2.y + M(1, 2);
                    double z = M(2, 0) * p2.x + M(2, 1) * p2.y + M(2, 2);
                    err_(2 * pos, 0) = p1.x - x / z;
                    err_(2 * pos + 1, 0) = p1.y - y / z;
                }
            }
        }


        void ReprojError_FixedK_OnlyR::Jacobian(const Mat &arg, Mat &jac) {
            Mat_<double> arg_(arg.clone());

            jac.create(dimension(), arg_.cols, CV_64F);
            Mat_<double> jac_(jac);
            jac_.setTo(0);

            // Maps argument index to the respective intrinsic parameter
            static const int flags_tbl[] = {REFINE_FLAG_FX, REFINE_FLAG_SKEW, REFINE_FLAG_PPX,
                                            REFINE_FLAG_FY, REFINE_FLAG_PPY};

            for (int i = 0; i < arg_.cols; ++i) {
                if (i > 4 || (params_to_refine_ & flags_tbl[i])) {
                    double val = arg_(0, i);

                    arg_(0, i) += step_;
                    Mat tmp = jac_.col(i);
                    (*this)(arg_, tmp);

                    arg_(0, i) = val - step_;
                    (*this)(arg_, err_);
                    arg_(0, i) = val;

                    for (int j = 0; j < dimension(); ++j)
                        jac_(j, i) = (jac_(j, i) - err_(j, 0)) / (2 * step_);
                }
            }
        }
    } // namespace


    double RefineRigidCamera(InputOutputArray K, AbsoluteRotationMats Rs,
                             const FeaturesCollection &features, const MatchesCollection &matches,
                             int params_to_refine)
    {
        CV_Assert(K.getMatRef().size() == Size(3, 3) && K.getMatRef().type() == CV_64F);
        Mat_<double> K_(K.getMatRef());

        // Normalize rotations and compute indices

        Mat R_norm = Rs.begin()->second.t();
        vector<int> Rs_indices;

        for (AbsoluteRotationMats::iterator iter = Rs.begin(); iter != Rs.end(); ++iter) {
            CV_Assert(iter->second.size() == Size(3, 3) && iter->second.type() == CV_64F);
            iter->second = R_norm * iter->second;
            Rs_indices.push_back(iter->first);
        }

        Mat_<double> arg(1, 5 + 3 * (int)Rs.size());
        arg(0, 0) = K_(0, 0);
        arg(0, 1) = K_(0, 1);
        arg(0, 2) = K_(0, 2);
        arg(0, 3) = K_(1, 1);
        arg(0, 4) = K_(1, 2);
        for (size_t i = 1; i < Rs_indices.size(); ++i) {
            Mat_<double> rvec;
            Rodrigues(Rs.find(Rs_indices[i])->second, rvec);
            arg(0, 5 + 3 * (i - 1)) = rvec(0, 0);
            arg(0, 5 + 3 * (i - 1) + 1) = rvec(0, 1);
            arg(0, 5 + 3 * (i - 1) + 2) = rvec(0, 2);
        }

        ReprojError_FixedK_OnlyR func(features, matches, params_to_refine, Rs_indices);
        double rms_error = MinimizeLevMarq(func, arg, MinimizeOpts::VERBOSE_SUMMARY);

        K_(0, 0) = arg(0, 0);
        K_(0, 1) = arg(0, 1);
        K_(0, 2) = arg(0, 2);
        K_(1, 1) = arg(0, 3);
        K_(1, 2) = arg(0, 4);
        for (size_t i = 1; i < Rs_indices.size(); ++i) {
            Mat_<double> rvec(1, 3);
            rvec(0, 0) = arg(0, 5 + 3 * (i - 1));
            rvec(0, 1) = arg(0, 5 + 3 * (i - 1) + 1);
            rvec(0, 2) = arg(0, 5 + 3 * (i - 1) + 2);
            Rodrigues(rvec, Rs.find(Rs_indices[i])->second);
        }

        return rms_error;
    }


    void AffineRectifyStereoCameraByTwoShots(
            InputOutputArray P_l, InputOutputArray P_r,
            InputOutputArray xy_l0, InputOutputArray xy_r0, InputOutputArray xy_l1, InputOutputArray xy_r1,
            const Ptr<vector<DMatch> > &matches_lr0, const Ptr<vector<DMatch> > &matches_lr1,
            const Ptr<vector<DMatch> > &matches_ll,
            OutputArray Hpa, OutputArray H01, OutputArray xyzw0, OutputArray xyzw1)
    {
        CV_Assert(P_l.getMat().type() == CV_64F && P_l.getMat().size() == Size(4, 3));
        CV_Assert(P_r.getMat().type() == CV_64F && P_r.getMat().size() == Size(4, 3));        

        CV_Assert(xy_l0.getMat().type() == CV_64F && xy_l0.getMat().rows == 1 && xy_l0.getMat().cols % 2 == 0);
        CV_Assert(xy_r0.getMat().type() == CV_64F && xy_r0.getMat().rows == 1 && xy_r0.getMat().cols % 2 == 0);
        CV_Assert(xy_l0.getMat().cols / 2 == xy_r0.getMat().cols / 2);

        CV_Assert(xy_l1.getMat().type() == CV_64F && xy_l1.getMat().rows == 1 && xy_l1.getMat().cols % 2 == 0);
        CV_Assert(xy_r1.getMat().type() == CV_64F && xy_r1.getMat().rows == 1 && xy_r1.getMat().cols % 2 == 0);
        CV_Assert(xy_l1.getMat().cols / 2 == xy_r1.getMat().cols / 2);

        Mat_<double> P_l_(P_l.getMat());
        Mat_<double> P_r_(P_r.getMat());
        Mat_<double> xy_l0_(xy_l0.getMat());
        Mat_<double> xy_r0_(xy_r0.getMat());
        Mat_<double> xy_l1_(xy_l1.getMat());
        Mat_<double> xy_r1_(xy_r1.getMat());

        // Find structure

        DltTriangulation dlt;

        Mat_<double> xyzw0_;
        dlt.triangulate(ProjectiveCamera(P_l_), ProjectiveCamera(P_r_), xy_l0_, xy_r0_, xyzw0_);

        Mat_<double> xyzw1_;
        dlt.triangulate(ProjectiveCamera(P_l_), ProjectiveCamera(P_r_), xy_l1_, xy_r1_, xyzw1_);

        AUTOCALIB_LOG(
            cout << "\nDLT reprojection RMS errors (l0 r0 l1 r1) = ("
                 << CalcRmsReprojectionError(xy_l0, P_l_, xyzw0_) << " "
                 << CalcRmsReprojectionError(xy_r0, P_r_, xyzw0_) << " "
                 << CalcRmsReprojectionError(xy_l1, P_l_, xyzw1_) << " "
                 << CalcRmsReprojectionError(xy_r1, P_r_, xyzw1_) << ")\n");

        // Leave only common part of point clouds

        vector<pair<int, int> > lr0_lr1_indices;
        Intersect(*matches_lr0, *matches_lr1, *matches_ll, lr0_lr1_indices);        

        Mat_<double> xy_l0_buf(1, lr0_lr1_indices.size() * 2);
        Mat_<double> xy_r0_buf(1, lr0_lr1_indices.size() * 2);
        Mat_<double> xy_l1_buf(1, lr0_lr1_indices.size() * 2);
        Mat_<double> xy_r1_buf(1, lr0_lr1_indices.size() * 2);
        Mat_<double> xyzw0_buf(1, lr0_lr1_indices.size() * 4);
        Mat_<double> xyzw1_buf(1, lr0_lr1_indices.size() * 4);

        for (size_t i = 0; i < lr0_lr1_indices.size(); ++i) {
            int i0 = lr0_lr1_indices[i].first;
            int i1 = lr0_lr1_indices[i].second;

            xy_l0_buf(0, 2 * i) = xy_l0_(0, 2 * i0);
            xy_l0_buf(0, 2 * i + 1) = xy_l0_(0, 2 * i0 + 1);

            xy_r0_buf(0, 2 * i) = xy_r0_(0, 2 * i0);
            xy_r0_buf(0, 2 * i + 1) = xy_r0_(0, 2 * i0 + 1);

            xy_l1_buf(0, 2 * i) = xy_l1_(0, 2 * i1);
            xy_l1_buf(0, 2 * i + 1) = xy_l1_(0, 2 * i1 + 1);

            xy_r1_buf(0, 2 * i) = xy_r1_(0, 2 * i1);
            xy_r1_buf(0, 2 * i + 1) = xy_r1_(0, 2 * i1 + 1);

            xyzw0_buf(0, 4 * i) = xyzw0_(0, 4 * i0);
            xyzw0_buf(0, 4 * i + 1) = xyzw0_(0, 4 * i0 + 1);
            xyzw0_buf(0, 4 * i + 2) = xyzw0_(0, 4 * i0 + 2);
            xyzw0_buf(0, 4 * i + 3) = xyzw0_(0, 4 * i0 + 3);

            xyzw1_buf(0, 4 * i) = xyzw1_(0, 4 * i1);
            xyzw1_buf(0, 4 * i + 1) = xyzw1_(0, 4 * i1 + 1);
            xyzw1_buf(0, 4 * i + 2) = xyzw1_(0, 4 * i1 + 2);
            xyzw1_buf(0, 4 * i + 3) = xyzw1_(0, 4 * i1 + 3);
        }

        xy_l0_ = xy_l0_buf;
        xy_r0_ = xy_r0_buf;
        xy_l1_ = xy_l1_buf;
        xy_r1_ = xy_r1_buf;
        xyzw0_ = xyzw0_buf;
        xyzw1_ = xyzw1_buf;

        // Find homography mapping the 1st cloud to the 2nd one

        int num_points_common = xyzw0_.cols / 4;

        AUTOCALIB_LOG(
            cout << "\nFinding H01 using " << num_points_common << " common points (point)...\n");

        Mat_<double> H01_ = FindHomographyLinear(xyzw0_, xyzw1_);

        Mat_<double> xyzw1_mapped(xyzw0_.size(), xyzw0_.type());
        for (int i = 0; i < num_points_common; ++i) {
            xyzw1_mapped(0, 4 * i) = H01_(0, 0) * xyzw0_(0, 4 * i) + H01_(0, 1) * xyzw0_(0, 4 * i + 1) + H01_(0, 2) * xyzw0_(0, 4 * i + 2) + H01_(0, 3) * xyzw0_(0, 4 * i + 3);
            xyzw1_mapped(0, 4 * i + 1) = H01_(1, 0) * xyzw0_(0, 4 * i) + H01_(1, 1) * xyzw0_(0, 4 * i + 1) + H01_(1, 2) * xyzw0_(0, 4 * i + 2) + H01_(1, 3) * xyzw0_(0, 4 * i + 3);
            xyzw1_mapped(0, 4 * i + 2) = H01_(2, 0) * xyzw0_(0, 4 * i) + H01_(2, 1) * xyzw0_(0, 4 * i + 1) + H01_(2, 2) * xyzw0_(0, 4 * i + 2) + H01_(2, 3) * xyzw0_(0, 4 * i + 3);
            xyzw1_mapped(0, 4 * i + 3) = H01_(3, 0) * xyzw0_(0, 4 * i) + H01_(3, 1) * xyzw0_(0, 4 * i + 1) + H01_(3, 2) * xyzw0_(0, 4 * i + 2) + H01_(3, 3) * xyzw0_(0, 4 * i + 3);
        }

        AUTOCALIB_LOG(
            cout << "Reprojection RMS error after mapping (l1 r1) = ("
                 << CalcRmsReprojectionError(xy_l1_, P_l_, xyzw1_mapped) << " "
                 << CalcRmsReprojectionError(xy_r1_, P_r_, xyzw1_mapped) << ")\n");

        // Finding plane-at-infinity

        AUTOCALIB_LOG(cout << "\nFinding plane-at-infinity...\n");

        cout << H01_ << endl;
        Mat_<double> p_inf = CalcPlaneAtInfinity(H01_);
        AUTOCALIB_LOG(cout << "Plane-at-infinity = " << p_inf << endl);

        // Affine rectification

        AUTOCALIB_LOG(cout << "\nAffine rectification...\n");

        Mat_<double> Hpa_ = Mat::eye(4, 4, CV_64F);
        Hpa_(3, 0) = -p_inf(0, 0); Hpa_(3, 1) = -p_inf(1, 0); Hpa_(3, 2) = -p_inf(2, 0);

        H01_ = Hpa_.inv() * H01_ * Hpa_;

        P_l_ = P_l_ * Hpa_;
        P_r_ = P_r_ * Hpa_;

        xyzw0_ = Hpa_.inv() * xyzw0_.reshape(num_points_common).t();
        xyzw1_ = Hpa_.inv() * xyzw1_.reshape(num_points_common).t();
        xyzw0_ = Mat(xyzw0_.t()).reshape(0, 1);
        xyzw1_ = Mat(xyzw1_.t()).reshape(0, 1);

        AUTOCALIB_LOG(
            cout << "Reprojection RMS error after affine rectification (l0 r0 l1 r1) = ("
                 << CalcRmsReprojectionError(xy_l0_, P_l_, xyzw0_) << " "
                 << CalcRmsReprojectionError(xy_r0_, P_r_, xyzw0_) << " "
                 << CalcRmsReprojectionError(xy_l1_, P_l_, xyzw1_) << " "
                 << CalcRmsReprojectionError(xy_r1_, P_r_, xyzw1_) << ")\n");

        P_l.getMatRef() = P_l_;
        P_r.getMatRef() = P_r_;
        xy_l0.getMatRef() = xy_l0_;
        xy_r0.getMatRef() = xy_r0_;
        xy_l1.getMatRef() = xy_l1_;
        xy_r1.getMatRef() = xy_r1_;
        Hpa.getMatRef() = Hpa_;
        H01.getMatRef() = H01_;
        xyzw0.getMatRef() = xyzw0_;
        xyzw1.getMatRef() = xyzw1_;
    }


    namespace {   

        // See details in Hartey R., Zisserman A., "Multiple View Geometry", 2nd ed., p. 287
        inline
        double SymEpipDist2(double x1, double y1, const Mat_<double> F12, double x2, double y2) {
            double x2_ = F12(0, 0) * x2 + F12(0, 1) * y2 + F12(0, 2);
            double y2_ = F12(1, 0) * x2 + F12(1, 1) * y2 + F12(1, 2);
            double z2_ = F12(2, 0) * x2 + F12(2, 1) * y2 + F12(2, 2);

            double x1_ = F12(0, 0) * x1 + F12(1, 0) * y1 + F12(2, 0);
            double y1_ = F12(0, 1) * x1 + F12(1, 1) * y1 + F12(2, 1);

            return Sqr(x1 * x2_ + y1 * y2_ + z2_) * (1 / (x1_ * x1_ + y1_ * y1_) +
                                                     1 / (x2_ * x2_ + y2_ * y2_));
        }

        class EpipError_FixedK_StereoCam {
        public:
            EpipError_FixedK_StereoCam(const FeaturesCollection &features,
                                       const MatchesCollection &matches,
                                       const vector<int> &motions_indices,
                                       int params_to_refine)
                : features_(&features), matches_(&matches), step_(1e-4),
                  params_to_refine_(params_to_refine)
            {
                num_matches_ = 0;
                for (MatchesCollection::const_iterator iter = matches_->begin();
                     iter != matches_->end(); ++iter)
                    num_matches_ += (int)iter->second->size();

                motions_indices_inv_.assign(*max_element(motions_indices.begin(), motions_indices.end()) + 1, -1);
                for (size_t i = 0; i < motions_indices.size(); ++i)
                    motions_indices_inv_[motions_indices[i]] = i;
            }

            void operator()(const Mat &arg, Mat &err);
            void Jacobian(const Mat &arg, Mat &jac);

            int dimension() const { return num_matches_; }

        private:
            const FeaturesCollection *features_;
            const MatchesCollection *matches_;
            int num_matches_;
            vector<int> motions_indices_inv_;
            int params_to_refine_;

            const double step_;
            Mat_<double> err_;
        };


        void EpipError_FixedK_StereoCam::operator()(const Mat &arg, Mat &err) {
            Mat_<double> arg_(arg);

            err.create(dimension(), 1, CV_64F);
            Mat_<double> err_(err);

            Mat_<double> K = Mat::eye(3, 3, CV_64F);
            K(0, 0) = arg_(0, 0);
            K(0, 1) = arg_(0, 1);
            K(0, 2) = arg_(0, 2);
            K(1, 1) = arg_(0, 3);
            K(1, 2) = arg_(0, 4);
            Mat K_inv = K.inv();

            Mat_<double> rvec_rel(1, 3);
            rvec_rel(0, 0) = arg_(0, 5);
            rvec_rel(0, 1) = arg_(0, 6);
            rvec_rel(0, 2) = arg_(0, 7);
            Mat R_rel;
            Rodrigues(rvec_rel, R_rel);

            Mat_<double> T_rel(3, 1);
            T_rel(0, 0) = arg_(0, 8);
            T_rel(1, 0) = arg_(0, 9);
            T_rel(2, 0) = arg_(0, 10);

            Mat_<double> F_rel = K_inv.t() * CrossProductMat(T_rel) * R_rel * K_inv;

            int pos = 0;
            for (MatchesCollection::const_iterator iter = matches_->begin();
                 iter != matches_->end(); ++iter)
            {
                int from = iter->first.first;
                int to = iter->first.second;

                const vector<KeyPoint> &kps_from = features_->find(from)->second->keypoints;
                const vector<KeyPoint> &kps_to = features_->find(to)->second->keypoints;

                if (BothAreLeft(from, to)) {
                    int from_ = from / 2;
                    int to_ = to / 2;

                    Mat_<double> rvec_from(1, 3);
                    if (motions_indices_inv_[from_] > 0) {
                        rvec_from(0, 0) = arg_(0, 11 + 6 * (motions_indices_inv_[from_] - 1));
                        rvec_from(0, 1) = arg_(0, 11 + 6 * (motions_indices_inv_[from_] - 1) + 1);
                        rvec_from(0, 2) = arg_(0, 11 + 6 * (motions_indices_inv_[from_] - 1) + 2);
                    }
                    else
                        rvec_from.setTo(0);
                    Mat R_from;
                    Rodrigues(rvec_from, R_from);

                    Mat_<double> rvec_to(1, 3);
                    if (motions_indices_inv_[to_] > 0) {
                        rvec_to(0, 0) = arg_(0, 11 + 6 * (motions_indices_inv_[to_] - 1));
                        rvec_to(0, 1) = arg_(0, 11 + 6 * (motions_indices_inv_[to_] - 1) + 1);
                        rvec_to(0, 2) = arg_(0, 11 + 6 * (motions_indices_inv_[to_] - 1) + 2);
                    }
                    else
                        rvec_to.setTo(0);
                    Mat R_to;
                    Rodrigues(rvec_to, R_to);

                    Mat_<double> T_from(3, 1);
                    if (motions_indices_inv_[from_] > 0) {
                        T_from(0, 0) = arg_(0, 11 + 6 * (motions_indices_inv_[from_] - 1) + 3);
                        T_from(1, 0) = arg_(0, 11 + 6 * (motions_indices_inv_[from_] - 1) + 4);
                        T_from(2, 0) = arg_(0, 11 + 6 * (motions_indices_inv_[from_] - 1) + 5);
                    }
                    else
                        T_from.setTo(0);

                    Mat_<double> T_to(3, 1);
                    if (motions_indices_inv_[to_] > 0) {
                        T_to(0, 0) = arg_(0, 11 + 6 * (motions_indices_inv_[to_] - 1) + 3);
                        T_to(1, 0) = arg_(0, 11 + 6 * (motions_indices_inv_[to_] - 1) + 4);
                        T_to(2, 0) = arg_(0, 11 + 6 * (motions_indices_inv_[to_] - 1) + 5);
                    }
                    else
                        T_to.setTo(0);

                    Mat R = R_to * R_from.t();
                    Mat_<double> F = K_inv.t() * CrossProductMat(R * T_from - T_to) * R * K_inv;

                    const vector<DMatch> &matches = *(iter->second);
                    for (size_t i = 0; i < matches.size(); ++i) {
                        const Point2f &p0 = kps_from[matches[i].queryIdx].pt;
                        const Point2f &p1 = kps_to[matches[i].trainIdx].pt;                       
                        err_(pos++, 0) = sqrt(SymEpipDist2(p1.x, p1.y, F, p0.x, p0.y));
                    }
                }
                else if (IsLeftRightPair(from, to)) {
                    const vector<DMatch> &matches = *(iter->second);
                    for (size_t i = 0; i < matches.size(); ++i) {
                        const Point2f &p0 = kps_from[matches[i].queryIdx].pt;
                        const Point2f &p1 = kps_to[matches[i].trainIdx].pt;
                        err_(pos++, 0) = sqrt(SymEpipDist2(p1.x, p1.y, F_rel, p0.x, p0.y));
                    }
                }
                else {
                    CV_Error(CV_StsError, "bad matches");
                }
            }
        }


        void EpipError_FixedK_StereoCam::Jacobian(const Mat &arg, Mat &jac) {
            Mat_<double> arg_(arg.clone());

            jac.create(dimension(), arg_.cols, CV_64F);
            Mat_<double> jac_(jac);
            jac_.setTo(0);

            // Maps argument index to the respective intrinsic parameter
            static const int flags_tbl[] = {REFINE_FLAG_FX, REFINE_FLAG_SKEW, REFINE_FLAG_PPX,
                                            REFINE_FLAG_FY, REFINE_FLAG_PPY};

            for (int i = 0; i < arg_.cols; ++i) {
                if (i > 4 || (params_to_refine_ & flags_tbl[i])) {
                    double val = arg_(0, i);

                    arg_(0, i) += step_;
                    Mat tmp = jac_.col(i);
                    (*this)(arg_, tmp);

                    arg_(0, i) = val - step_;
                    (*this)(arg_, err_);
                    arg_(0, i) = val;

                    for (int j = 0; j < dimension(); ++j)
                        jac_(j, i) = (jac_(j, i) - err_(j, 0)) / (2 * step_);
                }
            }
        }

    } // namespace


    double RefineStereoCamera(RigidCamera &cam, AbsoluteMotions motions,
                              const FeaturesCollection &features, const MatchesCollection &matches,
                              int params_to_refine)
    {
        // Normalize rotations and compute indices

        Mat R_norm = motions.begin()->second.R();
        Mat T_norm = motions.begin()->second.T();

        vector<int> motions_indices;

        for (AbsoluteMotions::iterator iter = motions.begin(); iter != motions.end(); ++iter) {
            iter->second.set_T(iter->second.T() - iter->second.R() * R_norm.t() * T_norm);
            iter->second.set_R(iter->second.R() * R_norm.t());
            motions_indices.push_back(iter->first);
        }

        Mat_<double> arg(1, 5/*K*/ + 3/*R*/ + 3/*T*/ + 6 * (int)motions.size());

        Mat_<double> K(cam.K());
        arg(0, 0) = K(0, 0);
        arg(0, 1) = K(0, 1);
        arg(0, 2) = K(0, 2);
        arg(0, 3) = K(1, 1);
        arg(0, 4) = K(1, 2);

        Mat_<double> rvec;
        Rodrigues(cam.R(), rvec);
        arg(0, 5) = rvec(0, 0);
        arg(0, 6) = rvec(0, 1);
        arg(0, 7) = rvec(0, 2);

        Mat_<double> T(cam.T());
        arg(0, 8) = T(0, 0);
        arg(0, 9) = T(1, 0);
        arg(0, 10) = T(2, 0);

        for (size_t i = 1; i < motions_indices.size(); ++i) {
            Mat_<double> rvec_l;
            Rodrigues(motions.find(motions_indices[i])->second.R(), rvec_l);
            arg(0, 11 + 6 * (i - 1)) = rvec_l(0, 0);
            arg(0, 11 + 6 * (i - 1) + 1) = rvec_l(0, 1);
            arg(0, 11 + 6 * (i - 1) + 2) = rvec_l(0, 2);

            Mat_<double> T_l = motions.find(motions_indices[i])->second.T();
            arg(0, 11 + 6 * (i - 1) + 3) = T_l(0, 0);
            arg(0, 11 + 6 * (i - 1) + 4) = T_l(1, 0);
            arg(0, 11 + 6 * (i - 1) + 5) = T_l(2, 0);
        }

        EpipError_FixedK_StereoCam func(features, matches, motions_indices, params_to_refine);
        double rms_error = MinimizeLevMarq(func, arg,
                                           MinimizeOpts(TermCriteria(TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                                                                     200, numeric_limits<double>::epsilon()), MinimizeOpts::VERBOSE_SUMMARY));

        K(0, 0) = arg(0, 0);
        K(0, 1) = arg(0, 1);
        K(0, 2) = arg(0, 2);
        K(1, 1) = arg(0, 3);
        K(1, 2) = arg(0, 4);

        rvec(0, 0) = arg(0, 5);
        rvec(0, 1) = arg(0, 6);
        rvec(0, 2) = arg(0, 7);

        T(0, 0) = arg(0, 8);
        T(1, 0) = arg(0, 9);
        T(2, 0) = arg(0, 10);

        Mat R;
        Rodrigues(rvec, R);
        cam = RigidCamera(K, R, T);

        for (size_t i = 1; i < motions_indices.size(); ++i) {
            Mat_<double> rvec_l(1, 3);
            rvec_l(0, 0) = arg(0, 11 + 6 * (i - 1));
            rvec_l(0, 1) = arg(0, 11 + 6 * (i - 1) + 1);
            rvec_l(0, 2) = arg(0, 11 + 6 * (i - 1) + 2);

            Mat R_l;
            Rodrigues(rvec_l, R_l);
            motions.find(motions_indices[i])->second.set_R(R_l);

            Mat_<double> T_l(3, 1);
            T_l(0, 0) = arg(0, 11 + 6 * (i - 1) + 3);
            T_l(1, 0) = arg(0, 11 + 6 * (i - 1) + 4);
            T_l(2, 0) = arg(0, 11 + 6 * (i - 1) + 5);
            motions.find(motions_indices[i])->second.set_T(T_l);
        }

        return rms_error;
    }


    void BestOf2NearestMatcher::match(const cv::detail::ImageFeatures &f1, 
                                      const cv::detail::ImageFeatures &f2,
                                      cv::detail::MatchesInfo &mi) 
    {
        vector<vector<DMatch> > matches;
        set<pair<int, int> > matches12;

        matcher_->knnMatch(f1.descriptors, f2.descriptors, matches, 2);
        for (size_t i = 0; i < matches.size(); ++i) {
            if (matches[i].size() < 2)
                continue;
            const DMatch &m1 = matches[i][0];
            const DMatch &m2 = matches[i][1];
            if (m1.distance < (1.f - match_conf_) * m2.distance)
                matches12.insert(make_pair(m1.queryIdx, m1.trainIdx));
        }

        mi.matches.clear();
        matcher_->knnMatch(f2.descriptors, f1.descriptors, matches, 2);
        for (size_t i = 0; i < matches.size(); ++i) {
            if (matches[i].size() < 2)
                continue;
            const DMatch &m1 = matches[i][0];
            const DMatch &m2 = matches[i][1];
            if (m1.distance < (1.f - match_conf_) * m2.distance &&
                matches12.find(make_pair(m1.trainIdx, m1.queryIdx)) != matches12.end())
            {
                mi.matches.push_back(DMatch(m1.trainIdx, m1.queryIdx, m1.distance));
            }
        }
    }


    void Intersect(const vector<DMatch> &matches_lr1, const vector<DMatch> &matches_lr2,
                   const vector<DMatch> &matches_ll, vector<pair<int, int> > &indices)
    {
        map<int, int> l1_to_lr1_idx;
        for (size_t i = 0; i < matches_lr1.size(); ++i)
            l1_to_lr1_idx.insert(make_pair(matches_lr1[i].queryIdx, i));

        map<int, int> l2_to_lr2_idx;
        for (size_t i = 0; i < matches_lr2.size(); ++i)
            l2_to_lr2_idx.insert(make_pair(matches_lr2[i].queryIdx, i));

        indices.clear();
        for (size_t i = 0; i < matches_ll.size(); ++i) {
            map<int, int>::iterator i1 = l1_to_lr1_idx.find(matches_ll[i].queryIdx);
            map<int, int>::iterator i2 = l2_to_lr2_idx.find(matches_ll[i].trainIdx);
            if (i1 != l1_to_lr1_idx.end() && i2 != l2_to_lr2_idx.end())
                indices.push_back(make_pair(i1->second, i2->second));
        }
    }


    Mat ExtractCameraMatFromFundamentalMat(InputArray F) {
        CV_Assert(F.getMat().type() == CV_64F && F.getMat().size() == Size(3, 3));
        Mat F_ = F.getMat();

        Mat epipole;
        SVD::solveZ(F_.t(), epipole);

        Mat P(3, 4, CV_64F);

        Mat A(P(Rect(0, 0, 3, 3)));

//        Mat v(1, 3, CV_64F);
//        RNG rng(0);

//        do {
//            rng.fill(v, RNG::UNIFORM, 0, 1);
            Mat(CrossProductMat(epipole) * F_ /*+ epipole * v*/).copyTo(A);
            A /= norm(A);
//        } while (abs(determinant(A)) < 1e-4);

        Mat a(P(Rect(3, 0, 1, 3)));
        epipole.copyTo(a);

        return P;
    }


    void DltTriangulation::triangulate(const IProjectiveCamera &P1, const IProjectiveCamera &P2,
                                       InputArray xy1, InputArray xy2, InputOutputArray xyzw)
    {
        CV_Assert(xy1.getMat().type() == CV_64F && xy1.getMat().rows == 1 && xy1.getMat().cols % 2 == 0);
        CV_Assert(xy2.getMat().type() == CV_64F && xy2.getMat().rows == 1 && xy2.getMat().cols % 2 == 0);
        CV_Assert(xy2.getMat().cols / 2 == xy2.getMat().cols / 2);

        Mat_<double> xy1_ = xy1.getMat().clone(), xy2_ = xy2.getMat().clone();
        int num_points = xy1_.cols / 2;

        Mat_<double> P1_ = P1.P(), P2_ = P2.P();
        P1_ /= norm(P1_); P2_ /= norm(P2_);

        // Normalize keypoints and cameras

        Mat_<double> T1 = CalcNormalizationMat3x3(xy1_);
        Mat_<double> T2 = CalcNormalizationMat3x3(xy2_);

        for (int i = 0; i < num_points; ++i) {
            xy1_(0, 2 * i) = T1(0, 0) * xy1_(0, 2 * i) + T1(0, 2);
            xy1_(0, 2 * i + 1) = T1(1, 1) * xy1_(0, 2 * i + 1) + T1(1, 2);
            xy2_(0, 2 * i) = T2(0, 0) * xy2_(0, 2 * i) + T2(0, 2);
            xy2_(0, 2 * i + 1) = T2(1, 1) * xy2_(0, 2 * i + 1) + T2(1, 2);
        }

        P1_ = T1 * P1_;
        P2_ = T2 * P2_;

        // Find points

        Mat &mat = xyzw.getMatRef();
        mat.create(1, 4 * num_points, CV_64F);
        Mat_<double> xyzw_(mat);

        Mat_<double> A(4, 4);

        for (int i = 0; i < num_points; ++i) {
            A.setTo(0);
            for (int j = 0; j < 4; ++j) {
                A(0, j) = xy1_(0, 2 * i) * P1_(2, j) - P1_(0, j);
                A(1, j) = xy1_(0, 2 * i + 1) * P1_(2, j) - P1_(1, j);
                A(2, j) = xy2_(0, 2 * i) * P2_(2, j) - P2_(0, j);
                A(3, j) = xy2_(0, 2 * i + 1) * P2_(2, j) - P2_(1, j);
            }

            // See http://stackoverflow.com/questions/2276445/triangulation-direct-linear-transform
            Mat(A.row(0)) /= norm(A.row(0));
            Mat(A.row(1)) /= norm(A.row(1));
            Mat(A.row(2)) /= norm(A.row(2));
            Mat(A.row(3)) /= norm(A.row(3));

            Mat sol;
            SVD::solveZ(A, sol);
            Mat_<double> pt = xyzw_.colRange(4 * i, 4 * (i + 1));
            Mat(sol.t()).copyTo(pt);
        }
    }


    Mat CalcNormalizationMat3x3(InputArray xy) {
        CV_Assert(xy.getMat().type() == CV_64F && xy.getMat().rows == 1 && xy.getMat().cols % 2 == 0);
        Mat_<double> xy_ = xy.getMat();
        int num_points = xy_.cols / 2;

        double cx = 0, cy = 0;
        for (int i = 0; i < num_points; ++i) {
            cx += xy_(0, 2 * i);
            cy += xy_(0, 2 * i + 1);
        }
        cx /= num_points;
        cy /= num_points;

        double mean_dist = 0;
        for (int i = 0; i < num_points; ++i) 
            mean_dist += sqrt(Sqr(cx - xy_(0, 2 * i)) + Sqr(cy - xy_(0, 2 * i + 1)));
        mean_dist /= num_points;

        double scale = num_points > 1 ? sqrt(2.) / mean_dist : 1;
        Mat_<double> T = Mat::eye(3, 3, CV_64F);
        T(0, 0) = scale; T(0, 2) = -cx * scale;
        T(1, 1) = scale; T(1, 2) = -cy * scale;
        
        return T;
    }


    double CalcRmsReprojectionError(InputArray xy, InputArray P, InputArray xyzw) {
        CV_Assert(xy.getMat().type() == CV_64F && xy.getMat().rows == 1 && xy.getMat().cols % 2 == 0);
        CV_Assert(P.getMat().type() == CV_64F && P.getMat().size() == Size(4, 3));
        CV_Assert(xyzw.getMat().type() == CV_64F && xyzw.getMat().rows == 1 && xyzw.getMat().cols % 4 == 0);
        CV_Assert(xy.getMat().cols / 2 == xyzw.getMat().cols / 4);

        Mat_<double> xy_ = xy.getMat();
        Mat_<double> P_ = P.getMat();
        Mat_<double> xyzw_ = xyzw.getMat();
        int num_points = xy_.cols / 2;

        double sum_sq_error = 0;
        double x, y, z;
        for (int i = 0; i < num_points; ++i) {            
            x = P_(0, 0) * xyzw_(0, 4 * i) + P_(0, 1) * xyzw_(0, 4 * i + 1) + P_(0, 2) * xyzw_(0, 4 * i + 2) + P_(0, 3) * xyzw_(0, 4 * i + 3);
            y = P_(1, 0) * xyzw_(0, 4 * i) + P_(1, 1) * xyzw_(0, 4 * i + 1) + P_(1, 2) * xyzw_(0, 4 * i + 2) + P_(1, 3) * xyzw_(0, 4 * i + 3);
            z = P_(2, 0) * xyzw_(0, 4 * i) + P_(2, 1) * xyzw_(0, 4 * i + 1) + P_(2, 2) * xyzw_(0, 4 * i + 2) + P_(2, 3) * xyzw_(0, 4 * i + 3);
            sum_sq_error += Sqr(xy_(0, 2 * i) - x / z) + Sqr(xy_(0, 2 * i + 1) - y / z);
        }

        return sqrt(sum_sq_error / num_points);
    }


    double CalcRmsEpipolarDistance(InputArray xy1, InputArray xy2, InputArray F) {
        CV_Assert(xy1.getMat().type() == CV_64F && xy1.getMat().rows == 1 && xy1.getMat().cols % 2 == 0);
        CV_Assert(xy2.getMat().type() == CV_64F && xy2.getMat().rows == 1 && xy2.getMat().cols % 2 == 0);
        CV_Assert(F.getMat().type() == CV_64F && F.getMat().size() == Size(3, 3));
        CV_Assert(xy1.getMat().cols / 2 == xy2.getMat().cols / 2);

        Mat_<double> xy1_ = xy1.getMat();
        Mat_<double> xy2_ = xy2.getMat();
        Mat_<double> F_ = F.getMat();
        int num_points = xy1_.cols / 2;

        double total_err = 0;
        for (int i = 0; i < num_points; ++i)
            total_err += SymEpipDist2(xy1_(0, 2 * i), xy1_(0, 2 * i + 1), F_,
                                      xy2_(0, 2 * i), xy2_(0, 2 * i + 1));

        return sqrt(total_err / num_points);
    }


    int FindFundamentalMatInliers(const detail::ImageFeatures &f1, const detail::ImageFeatures &f2,
                                  const vector<DMatch> &matches, InputArray F, double thresh,
                                  InputOutputArray mask)
    {
        CV_Assert(F.getMat().type() == CV_64F && F.getMat().size() == Size(3, 3));
        Mat_<double> F_(F.getMat());

        Mat &mask_tmp = mask.getMatRef();
        mask_tmp.create(1, matches.size(), CV_8U);
        mask_tmp.setTo(0);
        Mat_<uchar> mask_(mask_tmp);
        int num_inliers = 0;

        for (size_t i = 0; i < matches.size(); ++i) {
            const Point2f &p1 = f1.keypoints[matches[i].queryIdx].pt;
            const Point2f &p2 = f2.keypoints[matches[i].trainIdx].pt;

            double err = SymEpipDist2(p1.x, p1.y, F_, p2.x, p2.y);
            if (err < thresh * thresh) {
                mask_(0, i) = 1;
                num_inliers++;
            }
        }

        return num_inliers;
    }


    Mat FindHomographyLinear(InputArray xyzw1, InputArray xyzw2) {
        CV_Assert(xyzw1.getMat().type() == CV_64F && xyzw1.getMat().rows == 1 && xyzw1.getMat().cols % 4 == 0);
        CV_Assert(xyzw2.getMat().type() == CV_64F && xyzw2.getMat().rows == 1 && xyzw2.getMat().cols % 4 == 0);
        CV_Assert(xyzw1.getMat().cols / 4 == xyzw2.getMat().cols / 4);

        Mat_<double> xyzw1_ = xyzw1.getMat();
        Mat_<double> xyzw2_ = xyzw2.getMat();

        int num_points = xyzw1_.cols / 4;       
        CV_Assert(num_points >= 5); // TODO why 5?

        Mat_<double> A(6 * num_points, 16);
        A.setTo(0);

        /*
        x: matrix([x0], [x1], [x2], [x3]);
        y: matrix([y0], [y1], [y2], [y3]);
        H: matrix([h00,h01,h02,h03], [h10,h11,h12,h13],
                  [h20,h21,h22,h23], [h30,h31,h32,h33]);
        H1: matrix([0,1,0,0], [-(1),0,0,0], [-(0),-(0),0,0], [-(0),-(0),-(0),0]);
        H2: matrix([0,0,1,0], [-(0),0,0,0], [-(1),-(0),0,0], [-(0),-(0),-(0),0]);
        H3: matrix([0,0,0,1], [-(0),0,0,0], [-(0),-(0),0,0], [-(1),-(0),-(0),0]);
        H4: matrix([0,0,0,0], [-(0),0,1,0], [-(0),-(1),0,0], [-(0),-(0),-(0),0]);
        H5: matrix([0,0,0,0], [-(0),0,0,1], [-(0),-(0),0,0], [-(0),-(1),-(0),0]);
        H6: matrix([0,0,0,0], [-(0),0,0,0], [-(0),-(0),0,1], [-(0),-(0),-(1),0]);
        coefmatrix([transpose(y).H1.H.x=0, transpose(y).H2.H.x=0, transpose(y).H3.H.x=0,
                    transpose(y).H4.H.x=0, transpose(y).H5.H.x=0, transpose(y).H6.H.x=0],
                   [h00, h01, h02, h03, h10, h11, h12, h13,
                    h20, h21, h22, h23, h30, h31, h32, h33]);
        */

        static const int lut[][2] = {{1, 0}, {2, 0}, {3, 0}, {2, 1}, {3, 1}, {3, 2}};

        for (int p = 0; p < num_points; ++p) {
            double x[4] = {xyzw1_(0, 4 * p), xyzw1_(0, 4 * p + 1),
                           xyzw1_(0, 4 * p + 2), xyzw1_(0, 4 * p + 3)};
            double y[4] = {xyzw2_(0, 4 * p), xyzw2_(0, 4 * p + 1),
                           xyzw2_(0, 4 * p + 2), xyzw2_(0, 4 * p + 3)};
            for (int r = 0, c1 = 0; c1 < 3; ++c1) {
                for (int c2 = c1 + 1; c2 < 4; ++c2, ++r) {
                    for (int i = 0; i < 4; ++i) {
                        A(6 * p + r, 4 * c1 + i) = -x[i] * y[lut[r][0]];
                        A(6 * p + r, 4 * c2 + i) = x[i] * y[lut[r][1]];
                    }
                }
            }
        }

        for (int i = 0; i < A.rows; ++i)
            Mat(A.row(i)) /= norm(A.row(i)) + 1;

        Mat_<double> H;
        SVD::solveZ(A, H);
        H = H.reshape(4);

        return H / pow(abs(determinant(H)), 0.25);
    }


    Mat CalcPlaneAtInfinity(InputOutputArray H) {
        CV_Assert(H.getMat().type() == CV_64F && H.getMat().size() == Size(4, 4));
        Mat_<double> H_(H.getMat() / pow(abs(determinant(H)), 0.25));

        Mat_<double> evals1, evecs1;
        EigenDecompose(H_.t(), evals1, evecs1);

        cout << evecs1 << endl;
        cout << evals1 << endl;

        int best1 = 0;
        double min_dist1 = numeric_limits<double>::max();

        for (int i = 1; i < 4; ++i) {
            double dist = abs(evals1(0, 2 * i + 1));
            if (dist < min_dist1) {
                best1 = i;
                min_dist1 = dist;
            }
        }

        Mat_<double> evals2, evecs2;
        EigenDecompose(-H_.t(), evals2, evecs2);

        cout << evecs2 << endl;
        cout << evals2 << endl;

        int best2 = 0;
        double min_dist2 = numeric_limits<double>::max();

        for (int i = 1; i < 4; ++i) {
            double dist = abs(evals2(0, 2 * i + 1));
            if (dist < min_dist2) {
                best2 = i;
                min_dist2 = dist;
            }
        }

        vector<complex<double> > pinf(4);

        if (min_dist1 < min_dist2) {
            pinf[0] = evecs1.at<complex<double> >(best1, 0);
            pinf[1] = evecs1.at<complex<double> >(best1, 1);
            pinf[2] = evecs1.at<complex<double> >(best1, 2);
            pinf[3] = evecs1.at<complex<double> >(best1, 3);
        }
        else {
            pinf[0] = evecs2.at<complex<double> >(best2, 0);
            pinf[1] = evecs2.at<complex<double> >(best2, 1);
            pinf[2] = evecs2.at<complex<double> >(best2, 2);
            pinf[3] = evecs2.at<complex<double> >(best2, 3);
            H.getMatRef() = -H_;
        }

        pinf[0] /= pinf[3];
        pinf[1] /= pinf[3];
        pinf[2] /= pinf[3];
        pinf[3] = 1;

        cout << pinf[0] << " " << pinf[1] << " " << pinf[2] << " " << pinf[3] << endl;

        Mat_<double> pinf_real(4, 1);
        pinf_real(0, 0) = pinf[0].real();
        pinf_real(1, 0) = pinf[1].real();
        pinf_real(2, 0) = pinf[2].real();
        pinf_real(3, 0) = pinf[3].real();

        return pinf_real;
    }


    Mat FindFundamentalMatFromPairs(const FeaturesCollection &features, const MatchesCollection &matches,
                                    double thresh)
    {
        int num_matches = 0;
        for (MatchesCollection::const_iterator iter = matches.begin();
             iter != matches.end(); ++iter)
        {
            int from = iter->first.first;
            int to = iter->first.second;
            if (IsLeftRightPair(from, to))
                num_matches += (int)iter->second->size();
        }

        Mat_<double> xy1(1, num_matches * 2), xy2(1, num_matches * 2);

        int offset = 0;
        for (MatchesCollection::const_iterator iter = matches.begin();
             iter != matches.end(); ++iter)
        {
            int from = iter->first.first;
            int to = iter->first.second;

            if (!IsLeftRightPair(from, to))
                continue;

            const detail::ImageFeatures &f1 = *(features.find(from)->second);
            const detail::ImageFeatures &f2 = *(features.find(to)->second);

            Mat_<double> xy1_(xy1.colRange(2 * offset, 2 * (offset + (int)iter->second->size())));
            Mat_<double> xy2_(xy2.colRange(2 * offset, 2 * (offset + (int)iter->second->size())));

            ExtractMatchedKeypoints(f1, f2, *(iter->second), xy1_, xy2_);

            offset += (int)iter->second->size();
        }

        vector<uchar> F_mask;
        Mat F = findFundamentalMat(Mat(xy2).reshape(2), Mat(xy1).reshape(2), F_mask, FM_RANSAC,
                                   thresh);
//        Mat F = findFundamentalMat(Mat(xy2).reshape(2), Mat(xy1).reshape(2), F_mask, FM_LMEDS,
//                                   thresh);

        int num_inliers = 0;
        for (size_t i = 0; i < F_mask.size(); ++i) {
            if (F_mask[i])
                num_inliers++;
        }

        AUTOCALIB_LOG(
            cout << "#matches = " << num_matches
                 << ", #inliers = " << num_inliers
                 << ", RMS err = " << CalcRmsEpipolarDistance(xy1, xy2, F) << endl);

        return F;
    }


    Mat Antidiag(int rows, int cols, int type) {
        Mat dst = Mat::zeros(rows, cols, type);
        int len = min(rows, cols);

        switch (type) {
        case CV_8U:
            for (int i = 0; i < len; ++i)
                dst.at<uchar>(i, cols - i - 1) = 1;
            break;
        case CV_16S:
            for (int i = 0; i < len; ++i)
                dst.at<short>(i, cols - i - 1) = 1;
            break;
        case CV_32S:
            for (int i = 0; i < len; ++i)
                dst.at<int>(i, cols - i - 1) = 1;
            break;
        case CV_32F:
            for (int i = 0; i < len; ++i)
                dst.at<float>(i, cols - i - 1) = 1.f;
            break;
        case CV_64F:
            for (int i = 0; i < len; ++i)
                dst.at<double>(i, cols - i - 1) = 1;
            break;
        }

        return dst;
    }


    Mat DecomposeCholesky(InputArray mat) {
        Mat mat_ = mat.getMat();
        CV_Assert(mat_.rows == mat_.cols && mat_.type() == CV_64F);

        Mat L;
        mat_.copyTo(L);

        if (!Cholesky(L.ptr<double>(), L.step, L.cols, 0, 0, 0))
            return Mat();

        for (int i = 0; i < L.cols; ++i)
            for (int j = i + 1; j < L.rows; ++j)
                L.at<double>(i, j) = 0;

        for (int i = 0; i < L.cols; ++i)
            L.at<double>(i, i) = 1. / L.at<double>(i, i);

        return L;
    }


    Mat DecomposeUUt(InputArray mat) {
        Mat mat_ = mat.getMat();
        CV_Assert(mat_.rows == mat_.cols && mat_.type() == CV_64F);

        Mat adiag = Antidiag(3, 3, CV_64F);
        Mat U_flipped = DecomposeCholesky(adiag * mat_ * adiag);
        if (U_flipped.empty())
            return Mat();

        return adiag * U_flipped * adiag;
    }


    void ExtractMatchedKeypoints(const detail::ImageFeatures &f1, const detail::ImageFeatures &f2,
                                 const vector<DMatch> &matches, OutputArray xy1, OutputArray xy2)
    {
        Mat &xy1_ = xy1.getMatRef();
        Mat &xy2_ = xy2.getMatRef();

        xy1_.create(1, (int)matches.size() * 2, CV_64F);
        xy2_.create(1, (int)matches.size() * 2, CV_64F);

        for (size_t i = 0; i < matches.size(); ++i) {
            xy1_.at<Point2d>(0, i) = f1.keypoints[matches[i].queryIdx].pt;
            xy2_.at<Point2d>(0, i) = f2.keypoints[matches[i].trainIdx].pt;
        }
    }


    Point3d TransformRigid(const Point3d &point, const Mat &R, const Mat &T) {
        CV_Assert(R.size() == cv::Size(3, 3) && R.type() == CV_64F);
        CV_Assert(T.size() == cv::Size(1, 3) && T.type() == CV_64F);

        Mat_<double> R_(R), T_(T);
        Point3d result;
        result.x = R_(0, 0) * point.x + R_(0, 1) * point.y + R_(0, 2) * point.z + T_(0, 0);
        result.y = R_(1, 0) * point.x + R_(1, 1) * point.y + R_(1, 2) * point.z + T_(1, 0);
        result.z = R_(2, 0) * point.x + R_(2, 1) * point.y + R_(2, 2) * point.z + T_(2, 0);

        return result;
    }


    namespace {

        class IncrementDistance {
        public:
            IncrementDistance(map<int, int> &distances) : distances(&distances) {}

            void operator()(const detail::GraphEdge &edge) {
                (*distances)[edge.to] = (*distances)[edge.from] + 1;
            }

            map<int, int> *distances;
        };

    } // namespace


    int ExtractEfficientCorrespondences(int num_frames, const RelativeConfidences &rel_confs,
                                        detail::Graph &eff_corresp, RelativeConfidences *rel_confs_eff)
    {
        // Find connected components

        detail::DisjointSets cc_as_djs;
        cc_as_djs.createOneElemSets(num_frames);

        for (RelativeConfidences::const_iterator iter = rel_confs.begin(); iter != rel_confs.end(); ++iter) {
            int comp_from = cc_as_djs.findSetByElem(iter->first.first);
            int comp_to = cc_as_djs.findSetByElem(iter->first.second);
            if (comp_from != comp_to)
                cc_as_djs.mergeSets(comp_from, comp_to);
        }

        // Select the biggest one

        int max_comp_id = max_element(cc_as_djs.size.begin(), cc_as_djs.size.end())
                                      - cc_as_djs.size.begin();

        set<int> max_comp;
        for (int i = 0; i < num_frames; ++i)
            if (cc_as_djs.findSetByElem(i) == max_comp_id)
                max_comp.insert(i);

        // Leave only the biggest component data

        list<detail::GraphEdge> max_comp_edges;

        for (RelativeConfidences::const_iterator iter = rel_confs.begin(); iter != rel_confs.end(); ++iter) {
            if (max_comp.find(iter->first.first) != max_comp.end() &&
                max_comp.find(iter->first.second) != max_comp.end())
            {
                max_comp_edges.push_back(detail::GraphEdge(iter->first.first, iter->first.second, 
                                                           static_cast<float>(iter->second)));
            }
        }

        // Find a maximum spanning tree of the maximum component using the Kruskal algorithm

        detail::Graph &span_tree_bidirect = eff_corresp;
        span_tree_bidirect.create(num_frames);

        map<int, int> span_tree_powers;

        cc_as_djs.createOneElemSets(num_frames);
        max_comp_edges.sort(greater<detail::GraphEdge>());

        if (rel_confs_eff)
            rel_confs_eff->clear();

        for (list<detail::GraphEdge>::iterator iter = max_comp_edges.begin();
             iter != max_comp_edges.end(); ++iter)
        {
            int comp_from = cc_as_djs.findSetByElem(iter->from);
            int comp_to = cc_as_djs.findSetByElem(iter->to);

            if (comp_from != comp_to) {
                cc_as_djs.mergeSets(comp_from, comp_to);

                span_tree_bidirect.addEdge(iter->from, iter->to, iter->weight);
                span_tree_bidirect.addEdge(iter->to, iter->from, iter->weight);

                if (rel_confs_eff) {
                    double confidence = rel_confs.find(make_pair(iter->from, iter->to))->second;
                    (*rel_confs_eff)[make_pair(iter->from, iter->to)] = confidence;
                    (*rel_confs_eff)[make_pair(iter->to, iter->from)] = confidence;
                }

                map<int, int>::iterator iter_ = span_tree_powers.find(iter->from);
                if (iter_ != span_tree_powers.end())
                    iter_->second++;
                else
                    span_tree_powers[iter->from] = 1;

                iter_ = span_tree_powers.find(iter->to);
                if (iter_ != span_tree_powers.end())
                    iter_->second++;
                else
                    span_tree_powers[iter->to] = 1;
            }
        }

        // Find spanning tree leafs

        set<int> span_tree_leafs;
        map<int, int> zero_distances;

        for (map<int, int>::iterator iter = span_tree_powers.begin(); iter != span_tree_powers.end(); ++iter) {
            if (iter->second == 1) {
                span_tree_leafs.insert(iter->first);
                zero_distances.insert(make_pair(iter->first, 0));
            }
        }

        // Find spanning tree center

        int center;
        int radius = numeric_limits<int>::max();

        for (set<int>::iterator iter = max_comp.begin(); iter != max_comp.end(); ++iter) {
            map<int, int> distances = zero_distances;
            span_tree_bidirect.walkBreadthFirst(*iter, IncrementDistance(distances));

            int max_distance = numeric_limits<int>::min();

            for (map<int, int>::iterator iter_ = distances.begin(); iter_ != distances.end(); ++iter_) {
                if (iter_->second > max_distance)
                    max_distance = iter_->second;
            }

            if (max_distance < radius) {
                radius = max_distance;
                center = *iter;
            }
        }

        return center;
    }


    namespace {

        class CalcAbsoluteRotationMat {
        public:
            CalcAbsoluteRotationMat(const RelativeRotationMats &rel_rmats,
                               AbsoluteRotationMats &abs_rmats)
                : rel_rmats(&rel_rmats), abs_rmats(&abs_rmats) {}

            void operator()(const detail::GraphEdge &edge) {
                Mat R;

                RelativeRotationMats::const_iterator iter = rel_rmats->find(make_pair(edge.from, edge.to));
                if (iter != rel_rmats->end())
                    R = iter->second;
                else
                    R = rel_rmats->find(make_pair(edge.to, edge.from))->second.t();

                (*abs_rmats)[edge.to] = R * (*abs_rmats)[edge.from];
            }

            const RelativeRotationMats *rel_rmats;
            AbsoluteRotationMats *abs_rmats;
        };

    } // namespace


    void CalcAbsoluteRotations(const RelativeRotationMats &rel_rmats, const detail::Graph &eff_corresp,
                               int ref_frame_idx, AbsoluteRotationMats &abs_rmats)
    {
        abs_rmats.clear();
        abs_rmats[ref_frame_idx] = Mat::eye(3, 3, CV_64F);
        eff_corresp.walkBreadthFirst(ref_frame_idx, CalcAbsoluteRotationMat(rel_rmats, abs_rmats));
    }


    namespace {

        class CalcAbsoluteMotion {
        public:
            CalcAbsoluteMotion(const RelativeMotions &rel_motions, AbsoluteMotions &abs_motions)
                : rel_motions(&rel_motions), abs_motions(&abs_motions) {}

            void operator()(const detail::GraphEdge &edge) {
                Mat R, T;

                RelativeMotions::const_iterator iter = rel_motions->find(make_pair(edge.from, edge.to));
                if (iter != rel_motions->end()) {
                    R = iter->second.R();
                    T = iter->second.T();
                }
                else {
                    iter = rel_motions->find(make_pair(edge.to, edge.from));
                    R = iter->second.R().t();
                    T = -iter->second.R().t() * iter->second.T();
                }

                const Motion &motion_from = (*abs_motions)[edge.from];
                Motion &motion_to = (*abs_motions)[edge.to];

                motion_to.set_R(R * motion_from.R());
                motion_to.set_T(R * motion_from.T() + T);
            }

            const RelativeMotions *rel_motions;
            AbsoluteMotions *abs_motions;
        };

    } // namespace


    void CalcAbsoluteMotions(const RelativeMotions &rel_motions, const detail::Graph &eff_corresp,
                             int ref_idx, AbsoluteMotions &abs_motions)
    {
        abs_motions.clear();
        abs_motions[ref_idx] = Motion();
        eff_corresp.walkBreadthFirst(ref_idx, CalcAbsoluteMotion(rel_motions, abs_motions));
    }


    void EigenDecompose(InputArray mat, InputOutputArray vals, InputOutputArray vecs) {
        using namespace Eigen;

        CV_Assert(mat.getMat().type() == CV_64F && mat.getMat().rows == mat.getMat().cols);
        Mat_<double> mat_ = mat.getMat();
        int n = mat_.rows;

        MatrixXd eigen_mat(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                eigen_mat(i, j) = mat_(i, j);

        EigenSolver<MatrixXd> solver(eigen_mat);
        VectorXcd eigen_vals = solver.eigenvalues();
        MatrixXcd eigen_vecs = solver.eigenvectors();

        Mat &tmp_vals = vals.getMatRef();
        tmp_vals.create(1, n * 2, CV_64F);
        Mat_<double> vals_(tmp_vals);

        for (int i = 0; i < n; ++i) {
            vals_(0, 2 * i) = eigen_vals[i].real();
            vals_(0, 2 * i + 1) = eigen_vals[i].imag();
        }

        Mat &tmp_vecs = vecs.getMatRef();
        tmp_vecs.create(n, n * 2, CV_64F);
        Mat_<double> vecs_(tmp_vecs);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                vecs_(i, 2 * j) = eigen_vecs(j, i).real();
                vecs_(i, 2 * j + 1) = eigen_vecs(j, i).imag();
            }
        }
    }


    Mat CrossProductMat(InputArray vec) {
        CV_Assert(vec.getMat().type() == CV_64F && vec.getMat().size() == Size(1, 3));
        Mat_<double> vec_ = vec.getMat();

        Mat_<double> mat = Mat::zeros(3, 3, CV_64F);
        mat(0, 1) = -vec_(2, 0); mat(0, 2) = vec_(1, 0); mat(1, 2) = -vec_(0, 0);
        mat(1, 0) = -mat(0, 1); mat(2, 0) = -mat(0, 2); mat(2, 1) = -mat(1, 2);

        return mat;
    }

} // namespace autocalib


