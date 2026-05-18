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

#include <FairGBM/proxy_losses/proxy_loss_factory.hpp>
#include <FairGBM/proxy_losses/base.hpp>
#include <FairGBM/proxy_losses/hinge.hpp>
#include <FairGBM/proxy_losses/quadratic.hpp>
#include <FairGBM/proxy_losses/xentropy.hpp>
#include <FairGBM/config.h>
#include <LightGBM/meta.h>

#include <stdexcept>
#include <string>

namespace LightGBM {
namespace Constrained {

std::unique_ptr<ProxyLoss> ConstructProxyLoss(const FairGBM::Config& config) {
  std::string stepwise_proxy = config.constraint_stepwise_proxy;
  score_t margin = static_cast<score_t>(config.stepwise_proxy_margin);

  if (stepwise_proxy == "hinge") {
    return std::unique_ptr<HingeProxyLoss>(new HingeProxyLoss(margin));
  } else if (stepwise_proxy == "cross_entropy") {
    return std::unique_ptr<CrossEntropyProxyLoss>(new CrossEntropyProxyLoss(margin));
  } else if (stepwise_proxy == "quadratic") {
    return std::unique_ptr<QuadraticProxyLoss>(new QuadraticProxyLoss(margin));
  } else {
    throw std::invalid_argument("constraint_stepwise_proxy=" + stepwise_proxy + " not implemented!");
  }
}

}  // namespace Constrained
}  // namespace LightGBM
