#ifndef AUTOCALIB_CORE_INL_H_
#define AUTOCALIB_CORE_INL_H_

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core_c.h>
#include "core.h"

namespace autocalib {

template <typename Func>
double MinimizeLevMarq(Func func, cv::InputOutputArray arg, MinimizeOpts opts) {
    cv::Mat arg_ = arg.getMatRef();
    CV_Assert(arg_.type() == CV_64F && arg_.rows == 1 && arg_.isContinuous());   

    int err_dim = func.dimension();
    int arg_dim = arg_.cols;

    CvLevMarq solver(arg_dim, err_dim, opts.crit());
    CvMat arg_c = arg_.reshape(0, arg_dim);
    cvCopy(&arg_c, solver.param);

    double mean_err;
    cv::Mat err(err_dim, 1, CV_64F);
    cv::Mat jac(err_dim, arg_dim, CV_64F);

    func(arg_, err);
    double init_mean_err = err.dot(err) / func.dimension();

    int num_iters = 0;
    while (true) {
        const CvMat *solver_param = 0;
        CvMat *solver_err = 0;
        CvMat *solver_jac = 0;

        bool proceed = solver.update(solver_param, solver_jac, solver_err);
        cvCopy(solver_param, &arg_c);

        if (!proceed || !solver_err)
            break;

        if (solver_err) {
            func(arg_, err);
            CvMat tmp = err;
            cvCopy(&tmp, solver_err);
            mean_err = err.dot(err) / func.dimension();
        }

        if (solver_jac) {
            func.Jacobian(arg_, jac);
            CvMat tmp = jac;
            cvCopy(&tmp, solver_jac);
            num_iters++;
            if (opts.verbose() & MinimizeOpts::VerboseIter)
                LOG(std::cout << "MinimizeLevMarq: iter=" << num_iters
                              << ", mean_sq_err=" << mean_err << std::endl);
        }
    }

    if (opts.verbose() & MinimizeOpts::VerboseSummary)
        LOG(std::cout << "MinimizeLevMarq: start_mean_sq_err=" << init_mean_err
                      << ", final_mean_sq_err=" << mean_err << ", num_iters=" << num_iters << std::endl);

    return mean_err;
}

} // namespace autocalib

#endif // AUTOCALIB_CORE_INL_H_
