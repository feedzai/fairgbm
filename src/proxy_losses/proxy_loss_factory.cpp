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

#include <LightGBM/proxy_losses/proxy_loss_factory.hpp>
#include <LightGBM/proxy_losses/base.hpp>
#include <LightGBM/proxy_losses/hinge.hpp>
#include <LightGBM/proxy_losses/quadratic.hpp>
#include <LightGBM/proxy_losses/xentropy.hpp>
#include <LightGBM/config.h>

#include <string>

namespace LightGBM {
namespace Constrained {

std::unique_ptr<ProxyLoss> ConstructProxyLoss(const LightGBM::Config &config)
{
  std::string stepwise_proxy = config.constraint_stepwise_proxy;
  if (stepwise_proxy == "hinge") {
    return std::unique_ptr<HingeProxyLoss>(new HingeProxyLoss((score_t) config.stepwise_proxy_margin));
  }
  else if (stepwise_proxy == "cross_entropy")
  {
    return std::unique_ptr<CrossEntropyProxyLoss>(new CrossEntropyProxyLoss((score_t) config.stepwise_proxy_margin));
  }
  else if (stepwise_proxy == "quadratic")
  {
    return std::unique_ptr<QuadraticProxyLoss>(new QuadraticProxyLoss((score_t) config.stepwise_proxy_margin));
  }
  else {
    throw std::invalid_argument("constraint_stepwise_proxy=" + stepwise_proxy + " not implemented!");
  }
}

}   // Constrained
}   // LightGBM
