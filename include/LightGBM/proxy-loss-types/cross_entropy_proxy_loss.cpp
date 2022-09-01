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
#include "constrained.hpp"

#include <string>
#include <vector>
#include <unordered_map>

namespace LightGBM {

namespace Constrained {

class CrossEntropyProxyLoss : public ProxyLoss
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
        if (label_[i] == 0)
        {
            label_negatives[group] += 1;

            // proxy_margin_ corresponds to the vertical margin at x=0; l(0) = proxy_margin_
            const double xent_score = log(1 + exp(score[i] + xent_horizontal_shift_));
            assert(xent_score >= 0.);
            false_positives[group] += xent_score;
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
        if (label_[i] == 1)
        {
            label_positives[group] += 1;

            // proxy_margin_ corresponds to the vertical margin at x=0; l(0) = proxy_margin_
            const double xent_score = log(1 + exp(xent_horizontal_shift_ - score[i]));
            assert(xent_score >= 0.);
            false_negatives[group] += xent_score;
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
        return (Constrained::sigmoid(score + xent_horizontal_shift_)) / group_ln;
    }

    double ComputeInstancewiseFNR(const double score, const label_t label, const int group) const override
    {
        const int group_lp = group_label_positives_.at(group);
        return (Constrained::sigmoid(score - xent_horizontal_shift_) - 1) / group_lp;

    }

    double ComputeInstancewiseFPRGradient(const double score, const label_t label) const override
    {
        return (Constrained::sigmoid(score + xent_horizontal_shift_)) / total_label_negatives_;
    }

    double ComputeInstancewiseFNRGradient(const double score, const label_t label) const override
    {
        return (Constrained::sigmoid(score - xent_horizontal_shift_) - 1) / total_label_positives_;
    }
};

}
}
