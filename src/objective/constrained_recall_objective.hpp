/**
 * The copyright of this file belongs to Feedzai. The file cannot be
 * reproduced in whole or in part, stored in a retrieval system,
 * transmitted in any form, or by any means electronic, mechanical,
 * photocopying, or otherwise, without the prior permission of the owner.
 *
 * (c) 2021 Feedzai, Strictly Confidential
 */
/*!
 * Constrained proxy recall objective (minimize proxy FNR).
 */

#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"

#ifndef LIGHTGBM_CONSTRAINED_RECALL_OBJECTIVE_HPP
#define LIGHTGBM_CONSTRAINED_RECALL_OBJECTIVE_HPP

#include <LightGBM/meta.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/constrained.hpp>
#include "../metric/xentropy_metric.hpp"

#include <string>
#include <vector>

/**
 * Implements the proxy FNR loss (Recall as an objective).
 *
 * Could use BCE, Quadratic, or Exponential as proxy functions
 * for FNR's step-wise function.
 */

namespace LightGBM {

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
      std::cout << "constrained_recall_objective: boosting from scores == 0;" << std::endl;
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
       *  - This value should be zero in order to optimize solely for TPR (Recall),
       *  as TPR considers only label positives and ignores label negatives.
       *  - However, initial splits will have -inf information gain if the gradients
       *  of all label negatives are 0;
       *  - Hence, we're adding a small constant to the gradient of all LNs;
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
          // NOTE! trying to use a soft BCE signal for negative labels
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


}

#endif //LIGHTGBM_CONSTRAINED_RECALL_OBJECTIVE_HPP
