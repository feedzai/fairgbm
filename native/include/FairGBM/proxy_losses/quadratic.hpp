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

#ifndef FAIRGBM_PROXY_LOSSES_QUADRATIC_HPP_
#define FAIRGBM_PROXY_LOSSES_QUADRATIC_HPP_

#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>

namespace LightGBM {
namespace Constrained {

class QuadraticProxyLoss : public ProxyLoss {
 public:
  using ProxyLoss::ProxyLoss;

  /*! \brief virtual destructor */
  ~QuadraticProxyLoss() override = default;

  /**
   * Compute quadratic-proxy FPR (with a given margin).
   */
  inline double ComputeInstancewiseFPR(double score) const override {
    // LABEL is assumed to be NEGATIVE (0)
    return score >= -proxy_margin_ ? (1. / 2.) * std::pow(score + proxy_margin_, 2) : 0.;
  }

  /**
   * Compute quadratic-proxy FNR (with a given margin).
   */
  inline double ComputeInstancewiseFNR(double score) const override {
    // LABEL is assumed to be POSITIVE (1)
    return score <= proxy_margin_ ? (1. / 2.) * std::pow(score - proxy_margin_, 2) : 0.;
  }

  inline double ComputeInstancewiseFPRGradient(double score) const override {
    // LABEL is assumed to be NEGATIVE (0)
    return std::max(0., score + proxy_margin_);
  }

  inline double ComputeInstancewiseFNRGradient(double score) const override {
    // LABEL is assumed to be POSITIVE (1)
    return std::min(0., score - proxy_margin_);
  }
};

}  // namespace Constrained
}  // namespace LightGBM

#endif  // FAIRGBM_PROXY_LOSSES_QUADRATIC_HPP_
