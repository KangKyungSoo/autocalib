#include "precomp.h"
#include <include/core.h>

using namespace std;
using namespace cv;

namespace autocalib {

    Mat CalibRotationalCameraLinear(InputArrayOfArrays Hs) {
        vector<Mat> Hs_;
        Hs.getMatVector(Hs_);
        int num_Hs = (int)Hs_.size();
        if (num_Hs < 1)
            throw runtime_error("Need at least one homography");

        // Normalize homographies
        vector<Mat> Hs_normed(num_Hs);
        for (int i = 0; i < num_Hs; ++i) {
            CV_Assert(Hs_[i].size() == Size(3, 3) && Hs_[i].type() == CV_64F);
            Hs_normed[i] = Hs_[i] / pow(determinant(Hs_[i]), 1. / 3.);
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
                    eq_idx++;
                }
            }
        }

        Mat_<double> x;
        solve(A, b, x, DECOMP_SVD);
        Mat err = A * x - b;
        AUTOCALIB_LOG(cout << "solve() norm(A*x - b) / norm(b) = " << sqrt(err.dot(err) / b.dot(b)) << endl);

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


    Mat CalibRotationalCameraLinearNoSkew(InputArrayOfArrays Hs) {
        vector<Mat> Hs_;
        Hs.getMatVector(Hs_);
        int num_Hs = (int)Hs_.size();
        if (num_Hs < 1)
            throw runtime_error("Need at least one homography");

        // Normalize and transpose homographies
        vector<Mat> Hs_normed_t(num_Hs);
        for (int i = 0; i < num_Hs; ++i) {
            CV_Assert(Hs_[i].size() == Size(3, 3) && Hs_[i].type() == CV_64F);
            Hs_normed_t[i] = (Hs_[i] / pow(determinant(Hs_[i]), 1. / 3.)).t();
        }

        Mat_<double> A(6 * num_Hs, 4);
        Mat_<double> b(6 * num_Hs, 1);
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
                    eq_idx++;
                }
            }
        }

        Mat_<double> x;
        solve(A, b, x, DECOMP_SVD);
        Mat err = A * x - b;
        AUTOCALIB_LOG(cout << "solve() norm(A*x - b) / norm(b) = " << sqrt(err.dot(err) / b.dot(b)) << endl);

        // Image of the Absolute Conic == (K * K.t()).inv()
        Mat_<double> iac = Mat::eye(3, 3, CV_64F);
        iac(0, 0) = x(0, 0);
        iac(0, 2) = iac(2, 0) = x(1, 0);
        iac(1, 1) = x(2, 0);
        iac(1, 2) = iac(2, 1) = x(3, 0);

        AUTOCALIB_LOG(Mat evals; Mat evecs;
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

        class ReprojErrorFixedKR {
        public:
            ReprojErrorFixedKR(const FeaturesCollection &features,
                               const MatchesCollection &matches,
                               int params_to_refine,
                               const vector<int> &Rs_indices)
                    : features_(&features), matches_(&matches), params_to_refine_(params_to_refine),
                      step_(1e-4)
            {
                num_matches_ = 0;
                for (MatchesCollection::const_iterator view = matches_->begin();
                     view != matches_->end(); ++view)
                    num_matches_ += (int)view->second.size();

                Rs_indices_inv_.assign(*max(Rs_indices.begin(), Rs_indices.end()), -1);
                for (size_t i = 0; i < Rs_indices.size(); ++i)
                    Rs_indices_inv_[Rs_indices[i]] = i;
            }

            void operator()(const cv::Mat &arg, cv::Mat &err);
            void Jacobian(const cv::Mat &arg, cv::Mat &jac);

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


        void ReprojErrorFixedKR::operator()(const Mat &arg, Mat &err) {
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
                const vector<KeyPoint> &kps_from = features_->find(img_from)->second.keypoints;
                Mat_<double> rvec_from(1, 3);
                if (img_from) {
                    rvec_from(0, 0) = arg_(0, 5 + 3 * (Rs_indices_inv_[img_from] - 1));
                    rvec_from(0, 1) = arg_(0, 5 + 3 * (Rs_indices_inv_[img_from] - 1) + 1);
                    rvec_from(0, 2) = arg_(0, 5 + 3 * (Rs_indices_inv_[img_from] - 1) + 2);
                }
                else
                    rvec_from.setTo(0);
                Mat R_from;
                Rodrigues(rvec_from, R_from);

                int img_to = view->first.second;
                const vector<KeyPoint> &kps_to = features_->find(img_to)->second.keypoints;
                Mat_<double> rvec_to(1, 3);
                if (img_to) {
                    rvec_to(0, 0) = arg_(0, 5 + 3 * (Rs_indices_inv_[img_to] - 1));
                    rvec_to(0, 1) = arg_(0, 5 + 3 * (Rs_indices_inv_[img_to] - 1) + 1);
                    rvec_to(0, 2) = arg_(0, 5 + 3 * (Rs_indices_inv_[img_to] - 1) + 2);
                }
                else
                    rvec_to.setTo(0);
                Mat R_to;
                Rodrigues(rvec_to, R_to);

                Mat_<double> M = K * R_from * R_to.t() * K_inv;

                const vector<DMatch> &matches = view->second;
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


        // TODO calculate analytically Jacobian in BA
        void ReprojErrorFixedKR::Jacobian(const cv::Mat &arg, cv::Mat &jac) {
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


    void RefineRigidCamera(InputOutputArray K, AbsoluteRotationMats Rs,
                           const FeaturesCollection &features, const MatchesCollection &matches,
                           int params_to_refine)
    {
        CV_Assert(K.getMatRef().size() == Size(3, 3) && K.getMatRef().type() == CV_64F);
        Mat_<double> K_(K.getMatRef());

        vector<int> Rs_indices;
        for (AbsoluteRotationMats::iterator iter = Rs.begin(); iter != Rs.end() ;++iter) {
            CV_Assert(iter->second.size() == Size(3, 3) && iter->second.type() == CV_64F);
            iter->second = Rs.begin()->second.t() * iter->second;
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

        ReprojErrorFixedKR func(features, matches, params_to_refine, Rs_indices);
        MinimizeLevMarq(func, arg, MinimizeOpts::VERBOSE_SUMMARY);

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


    Mat DecomposeCholesky(InputArray src) {
        Mat src_ = src.getMat();
        CV_Assert(src_.rows == src_.cols && src_.type() == CV_64F);

        Mat L;
        src_.copyTo(L);

        if (!Cholesky(L.ptr<double>(), L.step, L.cols, 0, 0, 0))
            return Mat();

        for (int i = 0; i < L.cols; ++i)
            for (int j = i + 1; j < L.rows; ++j)
                L.at<double>(i, j) = 0;

        for (int i = 0; i < L.cols; ++i)
            L.at<double>(i, i) = 1. / L.at<double>(i, i);

        return L;
    }


    Mat DecomposeUUt(InputArray src) {
        Mat src_ = src.getMat();
        CV_Assert(src_.rows == src_.cols && src_.type() == CV_64F);

        Mat adiag = Antidiag(3, 3, CV_64F);
        Mat U_flipped = DecomposeCholesky(adiag * src_ * adiag);
        if (U_flipped.empty())
            return Mat();

        return adiag * U_flipped * adiag;
    }


    void ExtractMatchedKeypoints(const detail::ImageFeatures &f1, const detail::ImageFeatures &f2,
                                 const vector<DMatch> &matches, OutputArray kps1, OutputArray kps2)
    {
        Mat &kps1_ = kps1.getMatRef();
        Mat &kps2_ = kps2.getMatRef();

        kps1_.create(1, (int)matches.size(), CV_32FC2);
        kps2_.create(1, (int)matches.size(), CV_32FC2);

        for (size_t i = 0; i < matches.size(); ++i) {
            kps1_.at<Point2f>(0, i) = f1.keypoints[matches[i].queryIdx].pt;
            kps2_.at<Point2f>(0, i) = f2.keypoints[matches[i].trainIdx].pt;
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


    int ExtractAbsoluteRotations(const RelativeRotationMats &rel_rmats,
                                 const RelativeConfidences &rel_confs,
                                 AbsoluteRotationMats &abs_rmats)
    {
        set<int> vertices;

        // Prepare graph edges

        list<detail::GraphEdge> edges;
        for (RelativeRotationMats::const_iterator iter = rel_rmats.begin(); iter != rel_rmats.end(); ++iter) {
            vertices.insert(iter->first.first);
            vertices.insert(iter->first.second);
            double conf = rel_confs.find(iter->first)->second;
            if (conf > 0)
                edges.push_back(detail::GraphEdge(iter->first.first, iter->first.second, conf));
        }

        // Find maximum spanning subtrees using the Kruskal algorithm
        // (a few subtrees are possible as graph can be not connected)

        detail::DisjointSets comps_as_djs;
        edges.sort(greater<detail::GraphEdge>());
        for (list<detail::GraphEdge>::const_iterator iter = edges.begin(); iter != edges.end(); ++iter) {
            int comp_from = comps_as_djs.findSetByElem(iter->from);
            int comp_to = comps_as_djs.findSetByElem(iter->to);
            if (comp_from != comp_to)
                comps_as_djs.mergeSets(comp_from, comp_to);
        }

        // TODO finish ExtractAbsoluteRotations functions
        throw runtime_error("Not implemented");
    }

} // namespace autocalib
