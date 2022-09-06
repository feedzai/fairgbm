/**
 * The copyright of this file belongs to Feedzai. The file cannot be
 * reproduced in whole or in part, stored in a retrieval system,
 * transmitted in any form, or by any means electronic, mechanical,
 * photocopying, or otherwise, without the prior permission of the owner.
 *
 * (c) 2022 Feedzai, Strictly Confidential
 */

#ifndef LIGHTGBM_PROXY_LOSSES_BASE_HPP_
#define LIGHTGBM_PROXY_LOSSES_BASE_HPP_

#include <LightGBM/dataset.h>

#include <string>
#include <utility>
#include <vector>
#include <unordered_map>


namespace LightGBM {

namespace Constrained {

class ProxyLoss {
protected:
    /*! \brief Proxy margin */
    score_t proxy_margin_;

public:
    /*! \brief virtual destructor */
    virtual ~ProxyLoss() = default;

    explicit ProxyLoss(score_t proxy_margin) : proxy_margin_(proxy_margin) {};

    virtual void ComputeGroupwiseFPR(
            const double *score,
            std::unordered_map<constraint_group_t, double> &group_fpr,
            data_size_t num_data,
            const label_t *label,
            const label_t * /* weights */,
            const constraint_group_t *group,
            const std::vector<constraint_group_t> &group_values) const
    {
      std::unordered_map<constraint_group_t, double> false_positives; // map of group index to the respective proxy FPs
      std::unordered_map<constraint_group_t, int> label_negatives;    // map of group index to the respective number of LNs

      // #pragma omp parallel for schedule(static)        // TODO: https://github.com/feedzai/fairgbm/issues/6
      for (data_size_t i = 0; i < num_data; ++i)
      {
        constraint_group_t curr_group = group[i];

        // FPR uses only label NEGATIVES
        if (label[i] == 0)
        {
          label_negatives[curr_group] += 1;
          false_positives[curr_group] += this->ComputeInstancewiseFPR(score[i]);
        }
      }

      for (auto group_id : group_values)
      {
        double fpr;
        if (label_negatives[group_id] == 0)
          fpr = 0;
        else
          fpr = false_positives[group_id] / label_negatives[group_id];

        group_fpr[group_id] = fpr;
      }
    }

    virtual void ComputeGroupwiseFNR(
            const double *score,
            std::unordered_map<constraint_group_t, double> &group_fnr,
            data_size_t num_data,
            const label_t *label,
            const label_t * /* weights */,
            const constraint_group_t *group,
            const std::vector<constraint_group_t> &group_values) const
    {
      std::unordered_map<constraint_group_t, double> false_negatives; // map of group index to the respective proxy FPs
      std::unordered_map<constraint_group_t, int> label_positives;    // map of group index to the respective number of LNs

      // #pragma omp parallel for schedule(static)        // TODO: https://github.com/feedzai/fairgbm/issues/6
      for (data_size_t i = 0; i < num_data; ++i)
      {
        constraint_group_t curr_group = group[i];

        // FNR uses only label POSITIVES
        if (label[i] == 1)
        {
          label_positives[curr_group] += 1;
          false_negatives[curr_group] += this->ComputeInstancewiseFNR(score[i]);
        }
      }

      for (auto group_id : group_values)
      {
        double fnr;
        if (label_positives[group_id] == 0)
          fnr = 0;
        else
          fnr = false_negatives[group_id] / label_positives[group_id];

        group_fnr[group_id] = fnr;
      }
    }

    virtual double ComputeInstancewiseFPR(double score) const = 0;

    virtual double ComputeInstancewiseFNR(double score) const = 0;

    virtual double ComputeInstancewiseFPRGradient(double score) const = 0;

    virtual double ComputeInstancewiseFNRGradient(double score) const = 0;
};
}   // Constrained
}   // LightGBM

#endif  // LIGHTGBM_PROXY_LOSSES_BASE_HPP_
