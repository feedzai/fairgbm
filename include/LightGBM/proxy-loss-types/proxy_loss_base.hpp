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
#include <utility>
#include <vector>
#include <unordered_map>


namespace LightGBM {

namespace Constrained {

class ProxyLoss {
protected:
    /*! \brief Proxy margin */
    score_t proxy_margin_;

public:
    /*! \brief virtual destructor */
    virtual ~ProxyLoss() = default;

    explicit ProxyLoss(score_t proxy_margin) : proxy_margin_(proxy_margin) {};

    virtual void ComputeGroupwiseFPR(
            const double *score,
            std::unordered_map<constraint_group_t, double> &group_fpr,
            data_size_t num_data,
            const label_t *label,
            const label_t *weights,
            const constraint_group_t *group,
            const std::vector<constraint_group_t> &group_values) const = 0;

    virtual void ComputeGroupwiseFNR(
            const double *score,
            std::unordered_map<constraint_group_t, double> &group_fnr,
            data_size_t num_data,
            const label_t *label,
            const label_t *weights,
            const constraint_group_t *group,
            const std::vector<constraint_group_t> &group_values) const = 0;

    virtual double ComputeInstancewiseFPR(double score) const = 0;

    virtual double ComputeInstancewiseFNR(double score) const = 0;

    virtual double ComputeInstancewiseFPRGradient(double score) const = 0;

    virtual double ComputeInstancewiseFNRGradient(double score) const = 0;
};
}
}

#endif // PROXY_LOSS_BASE_HPP
