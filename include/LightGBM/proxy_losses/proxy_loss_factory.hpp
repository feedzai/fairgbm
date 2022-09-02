//
// Created by andre.cruz on 02/09/22.
//

#ifndef LIGHTGBM_PROXY_LOSSES_PROXY_LOSS_FACTORY_HPP_
#define LIGHTGBM_PROXY_LOSSES_PROXY_LOSS_FACTORY_HPP_

#include "base.hpp"
#include "hinge.hpp"
#include "xentropy.hpp"
#include "quadratic.hpp"

#include <string>

namespace LightGBM {
namespace Constrained {
    std::unique_ptr<ProxyLoss> ConstructProxyLoss(const Config &config)
    {
        std::string stepwise_proxy = config.constraint_stepwise_proxy;
        if (stepwise_proxy == "hinge") {
            return std::unique_ptr<HingeProxyLoss>(new HingeProxyLoss((score_t) config.stepwise_proxy_margin));
        }
//        else if (stepwise_proxy == "cross_entropy")
//        {
//            return std::unique_ptr<CrossEntropyProxyLoss>(new CrossEntropyProxyLoss(proxy_margin));
//        }
//        else if (stepwise_proxy == "quadratic")
//        {
//            return std::unique_ptr<QuadraticProxyLoss>(new QuadraticProxyLoss(proxy_margin));
//        }
        else {
            throw std::invalid_argument("constraint_stepwise_proxy=" + stepwise_proxy + " not implemented!");
        }
    }
}
}

#endif  // LIGHTGBM_PROXY_LOSSES_PROXY_LOSS_FACTORY_HPP_
