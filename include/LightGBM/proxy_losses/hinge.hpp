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
