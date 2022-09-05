/**
 * The copyright of this file belongs to Feedzai. The file cannot be
 * reproduced in whole or in part, stored in a retrieval system,
 * transmitted in any form, or by any means electronic, mechanical,
 * photocopying, or otherwise, without the prior permission of the owner.
 *
 * (c) 2022 Feedzai, Strictly Confidential
 */

#ifndef LIGHTGBM_PROXY_LOSSES_XENTROPY_HPP_
#define LIGHTGBM_PROXY_LOSSES_XENTROPY_HPP_

#include "base.hpp"
#include <LightGBM/dataset.h>
#include <LightGBM/utils/constrained.hpp>

#include <string>
#include <vector>
#include <unordered_map>

namespace LightGBM {
namespace Constrained {

class CrossEntropyProxyLoss : public ProxyLoss
{
private:
    /*! \brief Helper constant for BCE-based proxies */
    double xent_horizontal_shift_;

public:

    explicit CrossEntropyProxyLoss(score_t proxy_margin) : ProxyLoss(proxy_margin) {
      xent_horizontal_shift_ = log(exp(proxy_margin) - 1);
    };

    /*! \brief virtual destructor */
    ~CrossEntropyProxyLoss() override {
        std::cout << "DESTRUCTING CrossEntropyProxyLoss OBJECT !!" << std::endl; // TODO: delete this line, just for testing
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
        std::unordered_map<constraint_group_t, double> false_positives; // map of group index to the respective proxy FPs
        std::unordered_map<constraint_group_t, int> label_negatives;    // map of group index to the respective number of LNs

        // #pragma omp parallel for schedule(static)        // TODO: https://github.com/feedzai/fairgbm/issues/6
        for (data_size_t i = 0; i < num_data; ++i)
        {
            constraint_group_t curr_group = group[i];

            // FPR uses only label NEGATIVES
            if (label[i] == 0)
            {
                label_negatives[curr_group] += 1;

                // proxy_margin_ corresponds to the vertical margin at x=0; l(0) = proxy_margin_
                const double xent_score = log(1 + exp(score[i] + xent_horizontal_shift_));
                assert(xent_score >= 0.);
                false_positives[curr_group] += xent_score;
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
        std::unordered_map<constraint_group_t, double> false_negatives; // map of group index to the respective proxy FPs
        std::unordered_map<constraint_group_t, int> label_positives;    // map of group index to the respective number of LNs

        // #pragma omp parallel for schedule(static)        // TODO: https://github.com/feedzai/fairgbm/issues/6
        for (data_size_t i = 0; i < num_data; ++i)
        {
            constraint_group_t curr_group = group[i];

            // FNR uses only label POSITIVES
            if (label[i] == 1)
            {
                label_positives[curr_group] += 1;

                // proxy_margin_ corresponds to the vertical margin at x=0; l(0) = proxy_margin_
                const double xent_score = log(1 + exp(xent_horizontal_shift_ - score[i]));
                assert(xent_score >= 0.);
                false_negatives[curr_group] += xent_score;
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
//        return log(1 + exp(score + xent_horizontal_shift_));
    }

    inline double ComputeInstancewiseFNR(double score) const override
    {
        // LABEL is assumed to be POSITIVE (1)
//        return log(1 + exp(xent_horizontal_shift_ - score));
    }

    inline double ComputeInstancewiseFPRGradient(double score) const override
    {
        // LABEL is assumed to be NEGATIVE (0)
        return Constrained::sigmoid(score + xent_horizontal_shift_);
    }

    inline double ComputeInstancewiseFNRGradient(double score) const override
    {
        // LABEL is assumed to be POSITIVE (1)
        return Constrained::sigmoid(score - xent_horizontal_shift_) - 1;
    }
};

}   // Constrained
}   // LightGBM

#endif  // LIGHTGBM_PROXY_LOSSES_XENTROPY_HPP_
