#ifndef AUTOCALIB_CORE_H_
#define AUTOCALIB_CORE_H_

#include <vector>
#include <map>
#include <utility>
#include <opencv2/core/core.hpp>

namespace autocalib {

/**
  * Constructs anti-diagonal matrix of ones.
  *
  * \param rows Number of rows.
  * \param cols Number of cols.
  * \param type Matrix type.
  * \return Anti-diagonal matrix.
  */
cv::Mat Antidiag(int rows, int cols, int type);


/**
  * Performs Cholesky decomposition.
  *
  * \param src Symmetric positive-definite matrix (64F).
  * \param L Lower traingular matrix (64F), such as L * L.t() == src.
  * \return true if succeded, false otherwise.
  */
bool DecomposeCholesky(cv::InputArray src, cv::OutputArray L);


/** Calculates rotational camera intrinsics using linear algorithm.
  *
  * See details in Hartey R., Zisserman A., "Multiple View Geometry", 2nd ed., p. 482.
  *
  * \param Hs Homographies (64F).
  * \return Camera intrinsics (64F).
  */
cv::Mat CalibRotationalCameraLinear(cv::InputArrayOfArrays Hs);

} // namespace autocalib

#endif // AUTOCALIB_CORE_H_
