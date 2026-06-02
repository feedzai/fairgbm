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

#ifndef FAIRGBM_OBJECTIVE_CONSTRAINED_RECALL_OBJECTIVE_HPP_
#define FAIRGBM_OBJECTIVE_CONSTRAINED_RECALL_OBJECTIVE_HPP_

#include <FairGBM/constrained_objective_function.h>
#include <FairGBM/config.h>
#include <LightGBM/meta.h>
#include <FairGBM/utils/constrained.hpp>

#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace LightGBM {
namespace Constrained {

/**
 * Constrained proxy recall objective (minimize proxy FNR).
 * Minimizing FNR is equivalent to maximizing TPR (Recall), as TPR = 1-FNR.
 */
class ConstrainedRecallObjective : public ConstrainedObjectiveFunction {
 public:
  explicit ConstrainedRecallObjective(const FairGBM::Config& config) : deterministic_(config.deterministic) {
    SetUpFromConfig(config);

    if (!this->IsGlobalFPRConstrained())
      throw std::invalid_argument("Must provide a global FPR constraint in order to optimize for Recall!");

    if (objective_stepwise_proxy == "cross_entropy" || constraint_stepwise_proxy == "cross_entropy") {
      if (proxy_margin_ < DBL_MIN) {
        throw std::invalid_argument("Proxy margin must be positive.");
      }
    }

    if (objective_stepwise_proxy.empty()) {
      throw std::invalid_argument("Must provide an `objective_stepwise_proxy` to optimize for Recall.");
    }
  }

  // Constructor for model loading from string representation (resuming/prediction)
  explicit ConstrainedRecallObjective(const std::vector<std::string>&)
          : deterministic_(false) {
  }

  ~ConstrainedRecallObjective() override = default;

  const char* GetName() const override { return "constrained_recall_objective"; }

  std::string ToString() const override { return this->GetName(); }

  /**
   * Compute proxy FNR loss.
   *
   * Loss function:
   * - Quadratic: l(a) = (1/2) * (a - margin_)^2 * I[a < margin_],        where l(margin_) = 0
   * - BCE:       l(a) = log( 1 + exp( -a + log(exp(margin_) - 1) ) ),    where l(0) = margin_
   * - Hinge:     l(a) = (margin_ - a) * I[a < margin_],                  where l(margin_) = 0
   *
   * @param label The instance label.
   * @param score The instance predicted score.
   * @return The loss value.
   */
  double ComputePredictiveLoss(label_t label, double score) const override {
    if (std::abs(label) < 1e-5)
      return 0.;

    if (objective_stepwise_proxy == "quadratic") {
      return score < proxy_margin_ ? (1. / 2.) * std::pow(score - proxy_margin_, 2) : 0.;
    } else if (objective_stepwise_proxy == "cross_entropy") {
      double xent_horizontal_shift = std::log(std::exp(proxy_margin_) - 1);
      return std::log(1 + std::exp(-score + xent_horizontal_shift));
    } else if (objective_stepwise_proxy == "hinge") {
      return score < proxy_margin_ ? proxy_margin_ - score : 0.;
    } else {
      throw std::invalid_argument("Invalid objective_stepwise_proxy=" + objective_stepwise_proxy);
    }
  }

  /*!
   * The optimal constant-value model starts at logodds==0, as opposed to starting from the average score.
   * This is due using a different objective function, plus using global constraints.
   * @return 0
   */
  double BoostFromScore(int) const override {
    return 0.;
  }

  /**
   * > aka GetPredictiveLossGradientsWRTModelOutput
   *
   * Gradients of the proxy FNR loss w.r.t. the model output (scores).
   *
   * l(a) = (1/2) * (a - margin_)^2 * I[a < margin_]
   *
   * dl/da = (a - margin_) * I[a < margin_]
   *
   * @param score
   * @param gradients
   * @param hessians
   */
  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override {
    const double xent_horizontal_shift = std::log(std::exp(proxy_margin_) - 1);
    const double label_negative_weight = 1e-2;

    for (data_size_t i = 0; i < num_data_; ++i) {
      if (std::abs(label_[i] - 1) < 1e-5) {  // if (y_i == 1)
        if (objective_stepwise_proxy == "quadratic") {
          gradients[i] = static_cast<score_t>(score[i] < proxy_margin_ ? score[i] - proxy_margin_ : 0.);
          hessians[i] = static_cast<score_t>(score[i] < proxy_margin_ ? 1. : 0.);
        } else if (objective_stepwise_proxy == "cross_entropy") {
          const double z = Constrained::sigmoid(score[i] - xent_horizontal_shift);
          gradients[i] = static_cast<score_t>(z - 1.);
          hessians[i] = static_cast<score_t>(z * (1. - z));
        } else if (objective_stepwise_proxy == "hinge") {
          gradients[i] = static_cast<score_t>(score[i] < proxy_margin_ ? -1. : 0.);
          hessians[i] = static_cast<score_t>(0.);
        } else {
          throw std::invalid_argument("Invalid objective proxy: " + objective_stepwise_proxy);
        }

        if (weights_ != nullptr) {
          gradients[i] *= weights_[i];
          hessians[i] *= weights_[i];
        }

      } else {
        const double z = Constrained::sigmoid(score[i] + xent_horizontal_shift);
        gradients[i] = static_cast<score_t>(label_negative_weight * z);
        hessians[i] = static_cast<score_t>(label_negative_weight * z * (1. - z));
      }
    }
  }

  void GetConstraintGradientsWRTModelOutput(const double* multipliers, const double* score, score_t* gradients,
                                            score_t* hessians) const override {
    if (!this->IsGlobalFPRConstrained())
      throw std::invalid_argument("Recall objective function must have a global FPR constraint!");

    ConstrainedObjectiveFunction::GetConstraintGradientsWRTModelOutput(multipliers, score, gradients, hessians);
  }

 private:
  const bool deterministic_;
};
}  // namespace Constrained
}  // namespace LightGBM

#endif  // FAIRGBM_OBJECTIVE_CONSTRAINED_RECALL_OBJECTIVE_HPP_
