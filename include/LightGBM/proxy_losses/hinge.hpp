/**
 * The copyright of this file belongs to Feedzai. The file cannot be
 * reproduced in whole or in part, stored in a retrieval system,
 * transmitted in any form, or by any means electronic, mechanical,
 * photocopying, or otherwise, without the prior permission of the owner.
 *
 * (c) 2022 Feedzai, Strictly Confidential
 */

#ifndef LIGHTGBM_PROXY_LOSSES_HINGE_HPP_
#define LIGHTGBM_PROXY_LOSSES_HINGE_HPP_

#include "base.hpp"
#include <LightGBM/dataset.h>

#include <string>
#include <vector>
#include <unordered_map>

namespace LightGBM {

namespace Constrained {

class HingeProxyLoss : public ProxyLoss
{
public:

    using ProxyLoss::ProxyLoss;

    /*! \brief virtual destructor */
    ~HingeProxyLoss() override {
        std::cout << "DESTRUCTING HingeProxyLoss OBJECT !!" << std::endl; // TODO: delete this line, just for testing
    }

    void ComputeGroupwiseFPR(
            const double *score,
            std::unordered_map<constraint_group_t, double> &group_fpr,
            data_size_t num_data,
            const label_t *label,
            const label_t * /* weights */,
            const constraint_group_t *group,
            const std::vector<constraint_group_t> &group_values) const override
    {
        std::unordered_map<constraint_group_t, double> false_positives; // map of group index to the respective hinge-proxy FPs
        std::unordered_map<constraint_group_t, int> label_negatives;    // map of group index to the respective number of LNs

        // #pragma omp parallel for schedule(static)        // TODO: https://github.com/feedzai/fairgbm/issues/6
        for (data_size_t i = 0; i < num_data; ++i)
        {
        constraint_group_t curr_group = group[i];

        // HingeFPR uses only label negatives
        if (label[i] == 0)
        {
            label_negatives[curr_group] += 1;

            // proxy_margin_ is the line intercept value
            const double hinge_score = proxy_margin_ + score[i];
            false_positives[curr_group] += std::max(0.0, hinge_score);
        }
        }

        for (auto group_id : group_values)
        {
        double fpr;
        if (label_negatives[group_id] == 0)
            fpr = 0;
        else
            fpr = false_positives[group_id] / label_negatives[group_id];

        group_fpr[group_id] = fpr;
        }
    }

    void ComputeGroupwiseFNR(
            const double *score,
            std::unordered_map<constraint_group_t, double> &group_fnr,
            data_size_t num_data,
            const label_t *label,
            const label_t * /* weights */,
            const constraint_group_t *group,
            const std::vector<constraint_group_t> &group_values) const override
    {
        std::unordered_map<constraint_group_t, double> false_negatives; // map of group index to the respective hinge-proxy FNs
        std::unordered_map<constraint_group_t, int> label_positives;

        // #pragma omp parallel for schedule(static)        // TODO: https://github.com/feedzai/fairgbm/issues/6
        for (data_size_t i = 0; i < num_data; ++i)
        {
        constraint_group_t curr_group = group[i];

        if (label[i] == 1)
        {
            label_positives[curr_group] += 1;

            const double hinge_score = proxy_margin_ - score[i];
            false_negatives[curr_group] += std::max(0.0, hinge_score);
        }
        }

        for (auto group_id : group_values)
        {
        double fnr;
        if (label_positives[group_id] == 0)
            fnr = 0;
        else
            fnr = false_negatives[group_id] / label_positives[group_id];
        group_fnr[group_id] = fnr;
        }
    }

    inline double ComputeInstancewiseFPR(double score) const override
    {
        // LABEL is assumed to be NEGATIVE (0)
        return score <= -proxy_margin_ ? 0. : score + proxy_margin_;
    }

    inline double ComputeInstancewiseFNR(double score) const override
    {
        // LABEL is assumed to be POSITIVE (1)
        return score >= proxy_margin_ ? 0. : -score + proxy_margin_;
    }

    inline double ComputeInstancewiseFPRGradient(double score) const override
    {
        // LABEL is assumed to be NEGATIVE (0)
        return score >= -proxy_margin_ ? 1. : 0.;
    }

    inline double ComputeInstancewiseFNRGradient(double score) const override
    {
        // LABEL is assumed to be POSITIVE (1)
        return score >= proxy_margin_ ? 0. : -1.;
    }
};

}
}

#endif  // LIGHTGBM_PROXY_LOSSES_HINGE_HPP_
