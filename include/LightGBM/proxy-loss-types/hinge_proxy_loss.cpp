/**
 * The copyright of this file belongs to Feedzai. The file cannot be
 * reproduced in whole or in part, stored in a retrieval system,
 * transmitted in any form, or by any means electronic, mechanical,
 * photocopying, or otherwise, without the prior permission of the owner.
 *
 * (c) 2022 Feedzai, Strictly Confidential
 */

#include <LightGBM/dataset.h>
#include "proxy_loss_base.hpp"

#include <string>
#include <vector>
#include <unordered_map>

namespace LightGBM {

namespace Constrained {

class HingeProxyLoss : public ProxyLoss
{
public:
    HingeProxyLoss () = default;
    HingeProxyLoss (score_t proxy_margin, data_size_t num_data, label_t *label,  label_t *weights, const constraint_group_t *group, std::vector<constraint_group_t> group_values, std::unordered_map<constraint_group_t, int> group_label_positives, std::unordered_map<constraint_group_t, int> group_label_negatives, int total_label_negatives, int total_label_positives)
    {
        proxy_margin_ = proxy_margin;
        xent_horizontal_shift_ = log(exp(proxy_margin) - 1);
        num_data_ = num_data;
        label_ = label;
        weights_ = weights;
        group_ = group;
        group_values_ = group_values;
        group_label_positives_ = group_label_positives;
        group_label_negatives_ = group_label_negatives;
        total_label_negatives_ = total_label_negatives;
        total_label_positives_ = total_label_positives;
    };
    void ComputeGroupwiseFPR(const double *score, std::unordered_map<constraint_group_t, double> &group_fpr) const override
    {
        std::unordered_map<constraint_group_t, double> false_positives; // map of group index to the respective hinge-proxy FPs
        std::unordered_map<constraint_group_t, int> label_negatives;    // map of group index to the respective number of LNs

        // #pragma omp parallel for schedule(static)        // TODO: https://github.com/feedzai/fairgbm/issues/6
        for (data_size_t i = 0; i < num_data_; ++i)
        {
        constraint_group_t group = group_[i];

        // HingeFPR uses only label negatives
        if (label_[i] == 0)
        {
            label_negatives[group] += 1;

            // proxy_margin_ is the line intercept value
            const double hinge_score = proxy_margin_ + score[i];
            false_positives[group] += std::max(0.0, hinge_score);
        }
        }

        for (auto group_id : group_values_)
        {
        double fpr;
        if (label_negatives[group_id] == 0)
            fpr = 0;
        else
            fpr = false_positives[group_id] / label_negatives[group_id];

        group_fpr[group_id] = fpr;
        }
    }

    void ComputeGroupwiseFNR(const double *score, std::unordered_map<constraint_group_t, double> &group_fnr) const override
    {
        std::unordered_map<constraint_group_t, double> false_negatives; // map of group index to the respective hinge-proxy FNs
        std::unordered_map<constraint_group_t, int> label_positives;

        // #pragma omp parallel for schedule(static)        // TODO: https://github.com/feedzai/fairgbm/issues/6
        for (data_size_t i = 0; i < num_data_; ++i)
        {
        constraint_group_t group = group_[i];

        if (label_[i] == 1)
        {
            label_positives[group] += 1;

            const double hinge_score = proxy_margin_ - score[i];
            false_negatives[group] += std::max(0.0, hinge_score);
        }
        }

        for (auto group_id : group_values_)
        {
        double fnr;
        if (label_positives[group_id] == 0)
            fnr = 0;
        else
            fnr = false_negatives[group_id] / label_positives[group_id];
        group_fnr[group_id] = fnr;
        }
    }

    double ComputeInstancewiseFPR(const double score, const label_t label, const int group) const override
    {
        const int group_ln = group_label_negatives_.at(group);
        return score <= -proxy_margin_ ? 0. : 1. / group_ln;
    }

    double ComputeInstancewiseFNR(const double score, const label_t label, const int group) const override
    {
        const int group_lp = group_label_positives_.at(group);
        return score >= proxy_margin_ ? 0. : -1. / group_lp;
    }

    double ComputeInstancewiseFPRGradient(const double score, const label_t label) const override
    {
        return score >= -proxy_margin_ ? 1. / total_label_negatives_ : 0.;
    }

    double ComputeInstancewiseFNRGradient(const double score, const label_t label) const override
    {
        return score >= proxy_margin_ ? 0. : -1. / total_label_positives_;
    }
};

}
}
