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
#include <LightGBM/utils/constrained.hpp>

#include <string>
#include <vector>
#include <unordered_map>

namespace LightGBM {

namespace Constrained {

class CrossEntropyProxyLoss : public ProxyLoss
{
private:
    /*! \brief Helper constant for BCE-based proxies */
    double xent_horizontal_shift_;

    using ProxyLoss::ProxyLoss; // Will receive same args as parent class
    //         xent_horizontal_shift_ = log(exp(proxy_margin) - 1);     // TODO: initialize this var in the constructor

    /*! \brief virtual destructor */
    ~CrossEntropyProxyLoss() override {
        std::cout << "DESTRUCTING CrossEntropyProxyLoss OBJECT !!" << std::endl; // TODO: delete this line, just for testing
    }

    // TODO the rest
};

}
}
