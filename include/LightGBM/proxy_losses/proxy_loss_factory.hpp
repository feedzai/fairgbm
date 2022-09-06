/**
 * The copyright of this file belongs to Feedzai. The file cannot be
 * reproduced in whole or in part, stored in a retrieval system,
 * transmitted in any form, or by any means electronic, mechanical,
 * photocopying, or otherwise, without the prior permission of the owner.
 *
 * (c) 2022 Feedzai, Strictly Confidential
 */

#ifndef LIGHTGBM_PROXY_LOSSES_PROXY_LOSS_FACTORY_HPP_
#define LIGHTGBM_PROXY_LOSSES_PROXY_LOSS_FACTORY_HPP_

#include <LightGBM/proxy_losses/base.hpp>
#include <LightGBM/config.h>

#include <string>

namespace LightGBM {
namespace Constrained {

    std::unique_ptr<ProxyLoss> ConstructProxyLoss(const LightGBM::Config &config);

}   // Constrained
}   // LightGBM

#endif  // LIGHTGBM_PROXY_LOSSES_PROXY_LOSS_FACTORY_HPP_
