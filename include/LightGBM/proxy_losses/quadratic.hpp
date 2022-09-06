/**
 * The copyright of this file belongs to Feedzai. The file cannot be
 * reproduced in whole or in part, stored in a retrieval system,
 * transmitted in any form, or by any means electronic, mechanical,
 * photocopying, or otherwise, without the prior permission of the owner.
 *
 * (c) 2022 Feedzai, Strictly Confidential
 */

#ifndef LIGHTGBM_PROXY_LOSSES_QUADRATIC_HPP_
#define LIGHTGBM_PROXY_LOSSES_QUADRATIC_HPP_

#include "base.hpp"
#include <LightGBM/dataset.h>

#include <string>
#include <vector>
#include <unordered_map>

namespace LightGBM {
namespace Constrained {

class QuadraticProxyLoss : public ProxyLoss
{
public:

    using ProxyLoss::ProxyLoss;

    /*! \brief virtual destructor */
    ~QuadraticProxyLoss() override = default;

    /**
     * Compute quadratic-proxy FPR (with a given margin).
     *
     * Proxy FPR: (1/2) * (H_i + margin)^2 * I[H_i >= -margin and y_i == 0]
     *
     * proxy_margin_ corresponds to the symmetric of the function's zero point; f(-proxy_margin_)=0
     *
     * @param score array of scores
     * @param group_fpr hash-map of group to proxy-FPR
     */
    inline double ComputeInstancewiseFPR(double score) const override
    {
        // LABEL is assumed to be NEGATIVE (0)
        return score >= -proxy_margin_ ? (1. / 2.) * std::pow(score + proxy_margin_, 2) : 0.;
    }

    /**
     * Compute quadratic-proxy FNR (with a given margin).
     *
     * Proxy FNR: (1/2) * (H_i - margin)^2 * I[H_i <= margin and y_i == 1]
     *
     * proxy_margin_ corresponds to the function's zero point; f(proxy_margin_)=0
     *
     * @param score array of scores
     * @param group_fnr hash-map of group to proxy-FNR
     */
    inline double ComputeInstancewiseFNR(double score) const override
    {
        // LABEL is assumed to be POSITIVE (1)
        return score <= proxy_margin_ ? (1. / 2.) * std::pow(score - proxy_margin_, 2) : 0.;
    }

    inline double ComputeInstancewiseFPRGradient(double score) const override
    {
        // LABEL is assumed to be NEGATIVE (0)
        return std::max(0., score + proxy_margin_);
    }

    inline double ComputeInstancewiseFNRGradient(double score) const override
    {
        // LABEL is assumed to be POSITIVE (1)
        return std::min(0., score - proxy_margin_);
    }
};

}   // Constrained
}   // LightGBM

#endif  // LIGHTGBM_PROXY_LOSSES_QUADRATIC_HPP_
