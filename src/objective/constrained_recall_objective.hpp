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
/*!
 * Constrained proxy recall objective (minimize proxy FNR).
 */

#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"

#ifndef LIGHTGBM_OBJECTIVE_CONSTRAINED_RECALL_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_CONSTRAINED_RECALL_OBJECTIVE_HPP_

#include <LightGBM/meta.h>
#include <LightGBM/constrained_objective_function.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/constrained.hpp>
#include "../metric/xentropy_metric.hpp"

#include <string>
#include <vector>

/**
 * Implements the proxy FNR loss (Recall as an objective).
 *
 * Minimizing FNR is equivalent to maximizing TPR (or Recall), as TPR = 1-FNR.
 * Could use cross-entropy, quadratic, or hinge as proxy functions for FNR's step-wise function.
 * > We need to use a differentiable proxy, as the step-wise function provides no gradient for optimization.
 */

namespace LightGBM {
namespace Constrained {

class ConstrainedRecallObjective : public ConstrainedObjectiveFunction {
public:
    explicit ConstrainedRecallObjective(const Config &config)
            : deterministic_(config.deterministic) {
      SetUpFromConfig(config);

      if (not this->IsGlobalFPRConstrained())
        throw std::invalid_argument("Must provide a global FPR constraint in order to optimize for Recall!");

      if (objective_stepwise_proxy == "cross_entropy" or constraint_stepwise_proxy == "cross_entropy") {
        if (proxy_margin_ < DBL_MIN) {
          Log::Fatal("Proxy margin must be positive. It was %f.", proxy_margin_);
        }
      }

      if (objective_stepwise_proxy.empty()) {
        Log::Fatal("Must provide an `objective_stepwise_proxy` to optimize for Recall. Got empty input.");
      }

      // Disclaimer on using ConstrainedRecallObjective
      Log::Warning("Directly optimizing for Recall is still being researched and is prone to high variability of outcomes.");
    };

    explicit ConstrainedRecallObjective(const std::vector<std::string> &)
            : deterministic_(false) {
      throw std::invalid_argument(
              "I don't think this constructor should ever be called; "
              "it's only here for consistency with other objective functions.");
    }

    ~ConstrainedRecallObjective() override = default;

    const char *GetName() const override {
      return "constrained_recall_objective";
    }

    std::string ToString() const override {
      return this->GetName();
    }

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
      // If label is zero, loss will be zero
      if (abs(label) < 1e-5) // if (y_i == 0)
        return 0.;

      if (objective_stepwise_proxy == "quadratic")
        return score < proxy_margin_ ? (1./2.) * pow(score - proxy_margin_, 2) : 0.;  // proxy_margin_ is the HORIZONTAL margin!

      else if (objective_stepwise_proxy == "cross_entropy") {
        double xent_horizontal_shift = log(exp(proxy_margin_) - 1);       // proxy_margin_ is the VERTICAL margin!
        return log(1 + exp(-score + xent_horizontal_shift));
      }

      else if (objective_stepwise_proxy == "hinge")
        return score < proxy_margin_ ? proxy_margin_ - score : 0.;          // proxy_margin_ is the HORIZONTAL margin!

      else
        throw std::invalid_argument("Invalid objective_stepwise_proxy=" + objective_stepwise_proxy);
    }

    /*!
     * The optimal constant-value model starts at logodds==0, as opposed to starting from the average score.
     * This is due using a different objective function, plus using global constraints.
     * @return 0
     */
    double BoostFromScore(int) const override {
      Log::Info("constrained_recall_objective: boosting from scores == 0;");
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
    void GetGradients(const double *score, score_t *gradients, score_t *hessians) const override {
      /**
       * How much to shift the cross-entropy function (horizontally) to get
       * the target proxy_margin_ at x=0; i.e., f(0) = proxy_margin_
       */
      const double xent_horizontal_shift = log(exp(proxy_margin_) - 1);

      /**
       * NOTE
       *  - https://github.com/feedzai/fairgbm/issues/11
       *  - This value should be zero in order to optimize solely for TPR (Recall),
       *  as TPR considers only label positives (LPs) and ignores label negatives (LNs).
       *  - However, initial splits will have -inf information gain if the gradients
       *  of all LNs are 0;
       *  - Hence, we're adding a tiny positive weight to the gradient of all LNs;
       */
      const double label_negative_weight = 1e-2;

      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {

        // Proxy FNR (or proxy Recall) has no loss for label negative samples (they're ignored).
        if (abs(label_[i] - 1) < 1e-5) {  // if (y_i == 1)
          if (objective_stepwise_proxy == "quadratic") {
            gradients[i] = (score_t) (score[i] < proxy_margin_ ? score[i] - proxy_margin_ : 0.);
            hessians[i] = (score_t) (score[i] < proxy_margin_ ? 1. : 0.);
          }

          else if (objective_stepwise_proxy == "cross_entropy") {
            const double z = Constrained::sigmoid(score[i] - xent_horizontal_shift);
            gradients[i] = (score_t) (z - 1.);
            hessians[i] = (score_t) (z * (1. - z));
          }

          else if (objective_stepwise_proxy == "hinge") {
            gradients[i] = (score_t) (score[i] < proxy_margin_ ? -1. : 0.);
            hessians[i] = (score_t) 0.;
          }

          else {
            throw std::invalid_argument("Invalid objective proxy: " + objective_stepwise_proxy);
          }

          if (weights_ != nullptr) {
            gradients[i] *= weights_[i];
            hessians[i] *= weights_[i];
          }

        } else {
          // NOTE: https://github.com/feedzai/fairgbm/issues/11
          //  - This whole else clause should not be needed to optimize for Recall,
          //  as LNs have no influence on the FNR loss function or its (proxy-)gradient;
          //  - However, passing a zero gradient to all LNs leads to weird early stopping
          //  behavior from the `GBDT::Train` function;
          //  - Adding this tiny weight to the gradient of LNs seems to fix the issue with
          //  no (apparent) unintended consequences, as the gradient flowing is really small;
          const double z = Constrained::sigmoid(score[i] + xent_horizontal_shift);
          gradients[i] = (score_t) (label_negative_weight * z);
          hessians[i] = (score_t) (label_negative_weight * z * (1. - z));
        }
      }
    }

    void GetConstraintGradientsWRTModelOutput(const double *multipliers, const double *score, score_t *gradients,
                                              score_t *hessians) const override {
      if (not this->IsGlobalFPRConstrained())
        throw std::invalid_argument("Recall objective function must have a global FPR constraint!");

      ConstrainedObjectiveFunction::GetConstraintGradientsWRTModelOutput(multipliers, score, gradients, hessians);
    }

private:
    const bool deterministic_;

};
}   // namespace Constrained
}   // namespace LightGBM

#endif  // LIGHTGBM_OBJECTIVE_CONSTRAINED_RECALL_OBJECTIVE_HPP_
