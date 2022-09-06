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
    ~HingeProxyLoss() override = default;

    inline double ComputeInstancewiseFPR(double score) const override
    {
        // LABEL is assumed to be NEGATIVE (0)
        return std::max(0., score + proxy_margin_);
//        return score >= -proxy_margin_ ? score + proxy_margin_ : 0.;  // NOTE: equivalent notation
    }

    inline double ComputeInstancewiseFNR(double score) const override
    {
        // LABEL is assumed to be POSITIVE (1)
        return std::max(0., -score + proxy_margin_);
//        return score <= proxy_margin_ ? -score + proxy_margin_ : 0.;  // NOTE: equivalent notation
    }

    inline double ComputeInstancewiseFPRGradient(double score) const override
    {
        // LABEL is assumed to be NEGATIVE (0)
        return score >= -proxy_margin_ ? 1. : 0.;
    }

    inline double ComputeInstancewiseFNRGradient(double score) const override
    {
        // LABEL is assumed to be POSITIVE (1)
        return score <= proxy_margin_ ? -1. : 0.;
    }
};

}   // Constrained
}   // LightGBM

#endif  // LIGHTGBM_PROXY_LOSSES_HINGE_HPP_
