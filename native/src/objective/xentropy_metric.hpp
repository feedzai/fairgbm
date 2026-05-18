/*!
 * Copyright (c) 2017 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * Extracted helper functions used by constrained objectives.
 */
#ifndef FAIRGBM_OBJECTIVE_XENTROPY_METRIC_HPP_
#define FAIRGBM_OBJECTIVE_XENTROPY_METRIC_HPP_

#include <cmath>
#include <algorithm>
#include <LightGBM/meta.h>

namespace LightGBM {

  // label should be in interval [0, 1];
  // prob should be in interval (0, 1); prob is clipped if needed
  inline static double XentLoss(label_t label, double prob) {
    const double log_arg_epsilon = 1.0e-12;
    double a = label;
    if (prob > log_arg_epsilon) {
      a *= std::log(prob);
    } else {
      a *= std::log(log_arg_epsilon);
    }
    double b = 1.0f - label;
    if (1.0f - prob > log_arg_epsilon) {
      b *= std::log(1.0f - prob);
    } else {
      b *= std::log(log_arg_epsilon);
    }
    return - (a + b);
  }

}  // end namespace LightGBM

#endif  // FAIRGBM_OBJECTIVE_XENTROPY_METRIC_HPP_
