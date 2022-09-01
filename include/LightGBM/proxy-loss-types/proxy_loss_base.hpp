/**
 * The copyright of this file belongs to Feedzai. The file cannot be
 * reproduced in whole or in part, stored in a retrieval system,
 * transmitted in any form, or by any means electronic, mechanical,
 * photocopying, or otherwise, without the prior permission of the owner.
 *
 * (c) 2022 Feedzai, Strictly Confidential
 */

#ifndef PROXY_LOSS_BASE_HPP
#define PROXY_LOSS_BASE_HPP

#include <LightGBM/dataset.h>

#include <string>
#include <vector>
#include <unordered_map>


namespace LightGBM {

namespace Constrained {

class ProxyLoss {
protected:
    /*! \brief Proxy margin */
    score_t proxy_margin_;
    /*! \brief Helper constant for BCE-based proxies */
    double xent_horizontal_shift_;
    /*! \brief Number of data points */
    data_size_t num_data_;
    /*! \brief Pointer for label */
    const label_t *label_;
    /*! \brief Weights for data */
    const label_t *weights_;

    /*! \brief Pointer for group */
    const constraint_group_t *group_;
    /*! \brief Unique group values */
    std::vector<constraint_group_t> group_values_;

    /*! \brief Label positives per group */
    std::unordered_map<constraint_group_t, int> group_label_positives_;
    /*! \brief Label Negatives per group */
    std::unordered_map<constraint_group_t, int> group_label_negatives_;

    /*! \brief Total number of Label Negatives */
    int total_label_negatives_;
      /*! \brief Total number of Label Positives */
    int total_label_positives_;


public:
    ProxyLoss () = default;
    ProxyLoss (score_t proxy_margin, data_size_t num_data, label_t *label,  label_t *weights, const constraint_group_t *group, std::vector<constraint_group_t> group_values, std::unordered_map<constraint_group_t, int> group_label_positives, std::unordered_map<constraint_group_t, int> group_label_negatives, int total_label_negatives, int total_label_positives)
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

    virtual void ComputeGroupwiseFPR(const double *score, std::unordered_map<constraint_group_t, double> &group_fpr) const = 0;

    virtual void ComputeGroupwiseFNR(const double *score, std::unordered_map<constraint_group_t, double> &group_fnr) const = 0;

    virtual double ComputeInstancewiseFPR(const double score, const label_t label, const int group) const = 0;

    virtual double ComputeInstancewiseFNR(const double score, const label_t label, const int group) const = 0;

    virtual double ComputeInstancewiseFPRGradient(const double score, const label_t label) const = 0;

    virtual double ComputeInstancewiseFNRGradient(const double score, const label_t label) const = 0;
};

}
}

#endif // PROXY_LOSS_BASE_HPP
