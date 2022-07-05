/**
 * The copyright of this file belongs to Feedzai. The file cannot be
 * reproduced in whole or in part, stored in a retrieval system,
 * transmitted in any form, or by any means electronic, mechanical,
 * photocopying, or otherwise, without the prior permission of the owner.
 *
 * (c) 2022 Feedzai, Strictly Confidential
 */

#ifndef LIGHTGBM_LOSS_PROXIES_HPP
#define LIGHTGBM_LOSS_PROXIES_HPP

#include <LightGBM/dataset.h>
#include <LightGBM/meta.h>

#include <string>

namespace LightGBM {
namespace Constrained {

class ProxyLossFunction {
private:
    /*! \brief Number of data points */
    data_size_t num_data_;

    /*! \brief Pointer for label */
    const label_t *label_;

    /*! \brief Weights for data */
    const label_t *weights_;  // TODO: https://github.com/feedzai/fairgbm/issues/5

    /*! \brief Pointer for group */
    const constraint_group_t *group_;

    /*! \brief Unique group values */
    const std::vector<constraint_group_t> group_values_;

    /*! \brief Label positives per group */
    const std::unordered_map<constraint_group_t, int> group_label_positives_;

    /*! \brief Label Negatives per group */
    const std::unordered_map<constraint_group_t, int> group_label_negatives_;

    // Parameters
    /*! \brief Score threshold to compute confusion matrix (over predicted probabilities) */
    score_t score_threshold_ = 0.5;

    /*! \brief FPR threshold used in FPR constraints (small margin for constraint fulfillment) */
    score_t fpr_threshold_ = 0.0;

    /*! \brief FNR threshold used in FNR constraints (small margin for constraint fulfillment) */
    score_t fnr_threshold_ = 0.0;

    /*! \brief Margin threshold used in the Hinge approximation */
    score_t proxy_margin_ = 1.0;
    /*! \brief Target value for the global FPR constraint */
    score_t global_target_fpr_;

    /*! \brief Target value for the global FNR constraint */
    score_t global_target_fnr_;

    /*! \brief Score threshold used for the global constraints */
    score_t global_score_threshold_ = 0.5;

public:
    virtual ~ProxyLossFunction() {}

    virtual void Init(
            const Metadata& metadata,
            data_size_t num_data,
            const std::vector<constraint_group_t>& group_values,
            const std::unordered_map<constraint_group_t, int>& group_label_positives,
            const std::unordered_map<constraint_group_t, int>& group_label_negatives
            ) {
      num_data_ = num_data;
      label_ = metadata.label();
      weights_ = metadata.weights();
      group_ = metadata.constraint_group();
      group_values_(group_values);
      group_label_positives_(group_label_positives);
      group_label_negatives_(group_label_negatives);
    }

    void SetUpFromConfig(const Config &config) {
      score_threshold_ = (score_t)config.score_threshold;
      fpr_threshold_ = (score_t)config.constraint_fpr_threshold;
      fnr_threshold_ = (score_t)config.constraint_fnr_threshold;
      proxy_margin_ = (score_t)config.stepwise_proxy_margin;
      global_target_fpr_ = (score_t)config.global_target_fpr;
      global_target_fnr_ = (score_t)config.global_target_fnr;
      global_score_threshold_ = (score_t)config.global_score_threshold;
    }

    virtual const char* GetName() = 0;

    virtual double ComputeGroupwiseFPRLoss(const double* score, std::unordered_map<constraint_group_t, double> &group_fpr) const = 0;

    virtual double ComputeGroupwiseFNRLoss(const double* score, std::unordered_map<constraint_group_t, double> &group_fnr) const = 0;

    virtual double ComputeGlobalFPRLoss(const double* score) const = 0;

    virtual double ComputeGlobalFNRLoss(const double* score) const = 0;

    virtual double ComputeFPRGradientWRTPred(const double* score) const = 0;

    virtual double ComputeFNRGradientWRTPred(const double* score) const = 0;

    virtual double ComputeGlobalFPRGradientWRTPred(const double* score) const = 0;

    virtual double ComputeGlobalFNRGradientWRTPred(const double* score) const = 0;

};

}
}

#endif //LIGHTGBM_LOSS_PROXIES_HPP
