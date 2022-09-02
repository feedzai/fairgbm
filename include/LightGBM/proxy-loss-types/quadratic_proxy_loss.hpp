/**
 * The copyright of this file belongs to Feedzai. The file cannot be
 * reproduced in whole or in part, stored in a retrieval system,
 * transmitted in any form, or by any means electronic, mechanical,
 * photocopying, or otherwise, without the prior permission of the owner.
 *
 * (c) 2022 Feedzai, Strictly Confidential
 */

#include "proxy_loss_base.hpp"
#include <LightGBM/dataset.h>

#include <string>
#include <vector>
#include <unordered_map>

namespace LightGBM {

namespace Constrained {

class QuadraticProxyLoss : public ProxyLoss
{
    using ProxyLoss::ProxyLoss;

    /*! \brief virtual destructor */
    ~QuadraticProxyLoss() override {
        std::cout << "DESTRUCTING QuadraticProxyLoss OBJECT !!" << std::endl; // TODO: delete this line, just for testing
    }

    // TODO the rest
};

}
}
