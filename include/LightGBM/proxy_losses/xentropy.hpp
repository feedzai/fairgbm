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

    void ComputeGroupwiseFPR(const double *score, std::unordered_map<constraint_group_t, double> &group_fpr,
                             data_size_t num_data, const label_t *label, const label_t *weights,
                             const constraint_group_t *group,
                             const std::vector<constraint_group_t> &group_values) const override {

    }

    void ComputeGroupwiseFNR(const double *score, std::unordered_map<constraint_group_t, double> &group_fnr,
                             data_size_t num_data, const label_t *label, const label_t *weights,
                             const constraint_group_t *group,
                             const std::vector<constraint_group_t> &group_values) const override {

    }

    double ComputeInstancewiseFPR(double score) const override {
      return 0;
    }

    double ComputeInstancewiseFNR(double score) const override {
      return 0;
    }

    double ComputeInstancewiseFPRGradient(double score) const override {
      return 0;
    }

    double ComputeInstancewiseFNRGradient(double score) const override {
      return 0;
    }
};

}   // Constrained
}   // LightGBM

#endif  // LIGHTGBM_PROXY_LOSSES_XENTROPY_HPP_
