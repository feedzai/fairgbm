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

#ifndef FAIRGBM_PROXY_LOSSES_XENTROPY_HPP_
#define FAIRGBM_PROXY_LOSSES_XENTROPY_HPP_

#include "base.hpp"
#include <FairGBM/utils/constrained.hpp>

#include <string>
#include <vector>
#include <unordered_map>

namespace LightGBM {
namespace Constrained {

class CrossEntropyProxyLoss : public ProxyLoss {
 private:
  /*! \brief Helper constant for BCE-based proxies
   * proxy_margin_ corresponds to the vertical margin at score x=0; l(0) = proxy_margin_
   */
  const double xent_horizontal_shift_;

 public:
  explicit CrossEntropyProxyLoss(score_t proxy_margin)
      : ProxyLoss(proxy_margin), xent_horizontal_shift_(log(exp(proxy_margin) - 1)) {}

  /*! \brief virtual destructor */
  ~CrossEntropyProxyLoss() override = default;

  /**
   * Compute cross-entropy-proxy FPR.
   */
  inline double ComputeInstancewiseFPR(double score) const override {
    // LABEL is assumed to be NEGATIVE (0)
    return log(1 + exp(score + xent_horizontal_shift_));
  }

  /**
   * Compute cross-entropy-proxy FNR.
   */
  inline double ComputeInstancewiseFNR(double score) const override {
    // LABEL is assumed to be POSITIVE (1)
    return log(1 + exp(xent_horizontal_shift_ - score));
  }

  inline double ComputeInstancewiseFPRGradient(double score) const override {
    // LABEL is assumed to be NEGATIVE (0)
    return Constrained::sigmoid(score + xent_horizontal_shift_);
  }

  inline double ComputeInstancewiseFNRGradient(double score) const override {
    // LABEL is assumed to be POSITIVE (1)
    return Constrained::sigmoid(score - xent_horizontal_shift_) - 1;
  }
};

}  // namespace Constrained
}  // namespace LightGBM

#endif  // FAIRGBM_PROXY_LOSSES_XENTROPY_HPP_
