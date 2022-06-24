/**
 * The copyright of this file belongs to Feedzai. The file cannot be
 * reproduced in whole or in part, stored in a retrieval system,
 * transmitted in any form, or by any means electronic, mechanical,
 * photocopying, or otherwise, without the prior permission of the owner.
 *
 * (c) 2021 Feedzai, Strictly Confidential
 */
/*!
 * Copyright (c) 2017 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"

#ifndef LIGHTGBM_OBJECTIVE_CONSTRAINED_XENTROPY_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_CONSTRAINED_XENTROPY_OBJECTIVE_HPP_

#include <LightGBM/meta.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/constrained.hpp>
#include "../metric/xentropy_metric.hpp"

#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace LightGBM {

/**
 * Objective function for constrained optimization.
 * Uses the well-known Binary Cross Entropy (BCE) function for measuring predictive loss, plus
 * Uses a cross-entropy-based function as a proxy for the step-wise function when computing fairness constraints.
 *
 * NOTE:
 *  - This `constrained_xentropy` objective generally leads to the best constrained results;
 *  - All results from the FairGBM paper use this objective function with the "cross_entropy" step-wise proxy;
 *    - This pairing of "constrained cross-entropy objective + cross-entropy proxy for constraints" was tested the most;
 */
class ConstrainedCrossEntropy : public ConstrainedObjectiveFunction { // TODO: inherit from both CrossEntropy and ConstrainedObjectiveFunction
public:
  explicit ConstrainedCrossEntropy(const Config &config)
          : deterministic_(config.deterministic) {
    SetUpFromConfig(config);

    if (not objective_stepwise_proxy.empty()) {
      Log::Warning("Ignoring argument objective_stepwise_proxy=%s.", objective_stepwise_proxy.c_str());
    }
  }

  explicit ConstrainedCrossEntropy(const std::vector<std::string> &)
          : deterministic_(false) {
    Log::Warning(
            "The objective function 'constrained_cross_entropy' was not properly loaded. "
            "Resuming training is not available; everything else can be used as usual."
            );  // TODO: https://github.com/feedzai/fairgbm/issues/10
  }

  ~ConstrainedCrossEntropy() override = default;

  double ComputePredictiveLoss(label_t label, double score) const override {
    return XentLoss(label, Constrained::sigmoid(score));
  }

  /**
   * > aka GetPredictiveLossGradientsWRTModelOutput
   *
   * Gradient of the predictive loss w.r.t. model output (scores).
   * This is a duplicate of the implementation in the CrossEntropy class.
   *
   * @param score Model outputs.
   * @param gradients Reference to gradients' vector.
   * @param hessians Reference to hessians' vector.
   */
  void GetGradients(const double *score, score_t *gradients, score_t *hessians) const override {
    if (weights_ == nullptr) {
      // compute pointwise gradients and Hessians with implied unit weights
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double z = Constrained::sigmoid(score[i]);

        gradients[i] = static_cast<score_t>(z - label_[i]);     // 1st derivative
        hessians[i] = static_cast<score_t>(z * (1.0f - z));     // 2nd derivative
        // NOTE: should we set the 2nd derivative to zero? to stick to a 1st order method in both descent and ascent steps.
      }
    } else {
      // compute pointwise gradients and Hessians with given weights
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double z = Constrained::sigmoid(score[i]);

        gradients[i] = static_cast<score_t>((z - label_[i]) * weights_[i]);
        hessians[i] = static_cast<score_t>(z * (1.0f - z) * weights_[i]);
      }
    }
  }

  const char *GetName() const override {
    return "constrained_cross_entropy";
  }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    /* str_buf << "_->constraint_type->" << constraint_type;
    str_buf << "_->groups(";
    for (auto &group: group_values_)
      str_buf << group << ",";
    str_buf << ")";

    str_buf << "_score_threshold->" << score_threshold_;
    str_buf << "_fpr_threshold->" << fpr_threshold_;
    str_buf << "_fnr_threshold->" << fnr_threshold_; */
    return str_buf.str();
  }

  // implement custom average to boost from (if enabled among options)
  double BoostFromScore(int) const override {
    double suml = 0.0f;
    double sumw = 0.0f;
    if (weights_ != nullptr) {

      #pragma omp parallel for schedule(static) reduction(+:suml, sumw) if (!deterministic_)
      for (data_size_t i = 0; i < num_data_gr; ++i) {
        suml += label_[i] * weights_[i];
        sumw += weights_[i];
      }
    } else {
      sumw = static_cast<double>(num_data_);

      #pragma omp parallel for schedule(static) reduction(+:suml) if (!deterministic_)
      for (data_size_t i = 0; i < num_data_; ++i) {
        suml += label_[i];
      }
    }
    double pavg = suml / sumw;
    pavg = std::min(pavg, 1.0 - kEpsilon);
    pavg = std::max<double>(pavg, kEpsilon);
    double initscore = std::log(pavg / (1.0f - pavg));
    Log::Info("[%s:%s]: pavg = %f -> initscore = %f", GetName(), __func__, pavg, initscore);
    return initscore;
  }

private:
  const bool deterministic_;

};

}  // end namespace LightGBM

#endif   // end #ifndef LIGHTGBM_OBJECTIVE_CONSTRAINED_XENTROPY_OBJECTIVE_HPP_

#pragma clang diagnostic pop