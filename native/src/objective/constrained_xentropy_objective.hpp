/**
 * Copyright 2022 Feedzai
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef FAIRGBM_OBJECTIVE_CONSTRAINED_XENTROPY_OBJECTIVE_HPP_
#define FAIRGBM_OBJECTIVE_CONSTRAINED_XENTROPY_OBJECTIVE_HPP_

#include "xentropy_metric.hpp"
#include <FairGBM/constrained_objective_function.h>
#include <FairGBM/config.h>
#include <LightGBM/meta.h>
#include <FairGBM/utils/constrained.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace LightGBM {
namespace Constrained {

/**
 * Objective function for constrained optimization.
 * Uses the well-known Binary Cross Entropy (BCE) function for measuring
 * predictive loss, plus Uses a cross-entropy-based function as a proxy for the
 * step-wise function when computing fairness constraints.
 *
 * NOTE:
 *  - This `constrained_xentropy` objective generally leads to the best
 * constrained results;
 *  - All results from the FairGBM paper use this objective function with the
 * "cross_entropy" step-wise proxy;
 *    - This pairing of "constrained cross-entropy objective + cross-entropy
 * proxy for constraints" was tested the most;
 */
class ConstrainedCrossEntropy : public ConstrainedObjectiveFunction {
 public:
  explicit ConstrainedCrossEntropy(const FairGBM::Config& config) : deterministic_(config.deterministic) {
    SetUpFromConfig(config);
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
  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override {
    if (weights_ == nullptr) {
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double z = Constrained::sigmoid(score[i]);
        gradients[i] = static_cast<score_t>(z - label_[i]);
        hessians[i] = static_cast<score_t>(z * (1.0f - z));
      }
    } else {
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double z = Constrained::sigmoid(score[i]);
        gradients[i] = static_cast<score_t>((z - label_[i]) * weights_[i]);
        hessians[i] = static_cast<score_t>(z * (1.0f - z) * weights_[i]);
      }
    }
  }

  const char* GetName() const override { return "constrained_cross_entropy"; }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }

  double BoostFromScore(int) const override {
    double suml = 0.0f;
    double sumw = 0.0f;
    if (weights_ != nullptr) {
      for (data_size_t i = 0; i < num_data_; ++i) {
        suml += label_[i] * weights_[i];
        sumw += weights_[i];
      }
    } else {
      sumw = static_cast<double>(num_data_);
      for (data_size_t i = 0; i < num_data_; ++i) {
        suml += label_[i];
      }
    }
    double pavg = sumw > 0.0f ? suml / sumw : 0.0f;
    pavg = std::min(pavg, 1.0 - kEpsilon);
    pavg = std::max<double>(pavg, kEpsilon);
    return std::log(pavg / (1.0f - pavg));
  }

 private:
  const bool deterministic_;
};
}  // namespace Constrained
}  // namespace LightGBM

#endif  // FAIRGBM_OBJECTIVE_CONSTRAINED_XENTROPY_OBJECTIVE_HPP_
