/**
 * Copyright 2022 Feedzai
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
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
