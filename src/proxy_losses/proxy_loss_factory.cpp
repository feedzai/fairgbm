/**
 * The copyright of this file belongs to Feedzai. The file cannot be
 * reproduced in whole or in part, stored in a retrieval system,
 * transmitted in any form, or by any means electronic, mechanical,
 * photocopying, or otherwise, without the prior permission of the owner.
 *
 * (c) 2022 Feedzai, Strictly Confidential
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
