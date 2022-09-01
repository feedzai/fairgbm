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

class QuadraticProxyLoss : public ProxyLoss
{
    using ProxyLoss::ProxyLoss; // Will receive same args as parent class

    void ComputeGroupwiseFPR(const double *score, std::unordered_map<constraint_group_t, double> &group_fpr) const override
    {
        std::unordered_map<constraint_group_t, double> false_positives; // map of group index to the respective proxy FPs
        std::unordered_map<constraint_group_t, int> label_negatives;    // map of group index to the respective number of LNs

        // #pragma omp parallel for schedule(static)        // TODO: https://github.com/feedzai/fairgbm/issues/6
        for (data_size_t i = 0; i < num_data_; ++i)
        {
        constraint_group_t group = group_[i];

        // FPR uses only label NEGATIVES
        if (label_[i] == 0 and score[i] > -proxy_margin_)
        { // Conditions for non-zero proxy-FPR value
            label_negatives[group] += 1;

            // proxy_margin_ corresponds to the symmetric of the function's zero point; f(-proxy_margin_)=0
            const double quadratic_score = (1. / 2.) * std::pow(score[i] + proxy_margin_, 2);
            assert(quadratic_score >= 0.);
            false_positives[group] += quadratic_score;
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
        std::unordered_map<constraint_group_t, double> false_negatives; // map of group index to the respective proxy FPs
        std::unordered_map<constraint_group_t, int> label_positives;    // map of group index to the respective number of LNs

        // #pragma omp parallel for schedule(static)        // TODO: https://github.com/feedzai/fairgbm/issues/6
        for (data_size_t i = 0; i < num_data_; ++i)
        {
        constraint_group_t group = group_[i];

        // FNR uses only label POSITIVES
        if (label_[i] == 1 and score[i] < proxy_margin_)
        { // Conditions for non-zero proxy-FNR value
            label_positives[group] += 1;

            // proxy_margin_ corresponds to the function's zero point; f(proxy_margin_)=0
            const double quadratic_score = (1. / 2.) * std::pow(score[i] - proxy_margin_, 2);
            assert(quadratic_score >= 0.);
            false_negatives[group] += quadratic_score;
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
        return std::max(0., score + proxy_margin_) / group_ln;
    }

    double ComputeInstancewiseFNR(const double score, const label_t label, const int group) const override
    {
        const int group_lp = group_label_positives_.at(group);
        return std::min(0., score - proxy_margin_) / group_lp;
    }

    double ComputeInstancewiseFPRGradient(const double score, const label_t label) const override
    {
        return std::max(0., score + proxy_margin_) / total_label_negatives_;
    }

    double ComputeInstancewiseFNRGradient(const double score, const label_t label) const override
    {
        return std::min(0., score - proxy_margin_) / total_label_positives_;
    }
};

}
}
