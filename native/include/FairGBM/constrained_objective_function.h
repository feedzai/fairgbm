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

#ifndef FAIRGBM_CONSTRAINED_OBJECTIVE_FUNCTION_H_
#define FAIRGBM_CONSTRAINED_OBJECTIVE_FUNCTION_H_

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <FairGBM/compat.h>
#include <FairGBM/config.h>
#include <LightGBM/meta.h>
#include <LightGBM/objective_function.h>
#include <FairGBM/proxy_losses/base.hpp>
#include <FairGBM/proxy_losses/hinge.hpp>
#include <FairGBM/proxy_losses/proxy_loss_factory.hpp>
#include <FairGBM/proxy_losses/quadratic.hpp>
#include <FairGBM/proxy_losses/xentropy.hpp>
#include <FairGBM/utils/constrained.hpp>

namespace LightGBM {
namespace Constrained {

class ConstrainedObjectiveFunction : public ObjectiveFunction {
 public:
  enum constraint_type_t { FPR, FNR, FPR_AND_FNR, NONE, UNSET };

  /*! \brief virtual destructor */
  ~ConstrainedObjectiveFunction() override = default;

  void SetUpFromConfig(const FairGBM::Config& config) {
    constraint_type_str = config.constraint_type;

    // Normalize constraint type
    std::transform(constraint_type_str.begin(), constraint_type_str.end(), constraint_type_str.begin(), ::toupper);
    if (constraint_type_str == "FNR,FPR")
      constraint_type_str = "FPR,FNR";

    fpr_threshold_ = static_cast<score_t>(config.constraint_fpr_tolerance);
    fnr_threshold_ = static_cast<score_t>(config.constraint_fnr_tolerance);
    score_threshold_ = static_cast<score_t>(config.score_threshold);
    proxy_margin_ = static_cast<score_t>(config.stepwise_proxy_margin);

    /** Global constraint parameters **/
    global_constraint_type_str = config.global_constraint_type;

    // Normalize global constraint type
    std::transform(global_constraint_type_str.begin(), global_constraint_type_str.end(),
                   global_constraint_type_str.begin(), ::toupper);
    if (global_constraint_type_str == "FNR,FPR")
      global_constraint_type_str = "FPR,FNR";

    global_target_fpr_ = static_cast<score_t>(config.global_target_fpr);
    global_target_fnr_ = static_cast<score_t>(config.global_target_fnr);
    global_score_threshold_ = static_cast<score_t>(config.global_score_threshold);

    // Function used as a PROXY for step-wise in the CONSTRAINTS
    constraint_stepwise_proxy = ValidateProxyFunctionName(config.constraint_stepwise_proxy, false);

    // Function used as a PROXY for the step-wise in the OBJECTIVE
    objective_stepwise_proxy = ValidateProxyFunctionName(config.objective_stepwise_proxy, true);

    // Debug configs
    debugging_output_dir_ = config.debugging_output_dir;

    // Construct ProxyLoss object for constraint functions
    constraint_proxy_object = ConstructProxyLoss(config);

    // Set type of GROUP constraints
    if (constraint_type_str == "FPR") {
      group_constraint = FPR;
    } else if (constraint_type_str == "FNR") {
      group_constraint = FNR;
    } else if (constraint_type_str == "FPR,FNR") {
      group_constraint = FPR_AND_FNR;
    } else {
      group_constraint = NONE;
    }

    // Set type of GLOBAL constraints
    if (global_constraint_type_str == "FPR") {
      global_constraint = FPR;
    } else if (global_constraint_type_str == "FNR") {
      global_constraint = FNR;
    } else if (global_constraint_type_str == "FPR,FNR") {
      global_constraint = FPR_AND_FNR;
    } else {
      global_constraint = NONE;
    }
  }

  /*!
   * \brief Initialize — not supported in standalone FairGBM build.
   * Use FairGBM_ObjectiveInit (C API) instead.
   */
  void Init(const Metadata& /*metadata*/, data_size_t /*num_data*/) override {
    throw std::runtime_error(
        "ConstrainedObjectiveFunction::Init(Metadata&) is not supported in "
        "standalone FairGBM. Use the C API (FairGBM_ObjectiveInit) instead.");
  }

  /**
   * Template method for computing an instance's predictive loss value.
   */
  virtual double ComputePredictiveLoss(label_t label, double score) const = 0;

  /*!
   * \brief Get Lagrangian gradients w.r.t. multipliers (constraint violation values).
   */
  virtual std::vector<double> GetLagrangianGradientsWRTMultipliers(const double* score) const {
    if (weights_ != nullptr)
      throw std::logic_error("Weighted constraint violations not implemented yet");

    std::vector<double> constraint_values;
    std::unordered_map<constraint_group_t, double> group_fpr, group_fnr;

    // Multiplier corresponding to group-wise FPR constraints
    if (IsGroupFPRConstrained()) {
      ComputeFPR(score, score_threshold_, group_fpr);
      double max_fpr = Constrained::findMaxValuePair<constraint_group_t, double>(group_fpr).second;

      for (const auto& group : group_values_) {
        double fpr_constraint_value = max_fpr - group_fpr[group] - fpr_threshold_;
        constraint_values.push_back(fpr_constraint_value);
      }
    }

    // Multiplier corresponding to group-wise FNR constraints
    if (IsGroupFNRConstrained()) {
      ComputeFNR(score, score_threshold_, group_fnr);
      double max_fnr = Constrained::findMaxValuePair<constraint_group_t, double>(group_fnr).second;

      for (const auto& group : group_values_) {
        double fnr_constraint_value = max_fnr - group_fnr[group] - fnr_threshold_;
        constraint_values.push_back(fnr_constraint_value);
      }
    }

    // Global FPR constraint
    if (IsGlobalFPRConstrained()) {
      double global_fpr = ComputeGlobalFPR(score, global_score_threshold_);
      constraint_values.push_back(global_fpr - global_target_fpr_);
    }

    // Global FNR constraint
    if (IsGlobalFNRConstrained()) {
      double global_fnr = ComputeGlobalFNR(score, global_score_threshold_);
      constraint_values.push_back(global_fnr - global_target_fnr_);
    }

    return constraint_values;
  }

  /*!
   * \brief Get constraint gradients w.r.t. model output (uses proxy constraints).
   */
  virtual void GetConstraintGradientsWRTModelOutput(const double* lagrangian_multipliers, const double* score,
                                                    score_t* gradients, score_t* /* hessians */) const {
    std::unordered_map<constraint_group_t, double> group_fpr, group_fnr;
    std::pair<constraint_group_t, double> max_proxy_fpr, max_proxy_fnr;

    if (IsGroupFPRConstrained()) {
      constraint_proxy_object->ComputeGroupwiseFPR(score, group_fpr, num_data_, label_, weights_, group_,
                                                   group_values_);
      max_proxy_fpr = Constrained::findMaxValuePair<constraint_group_t, double>(group_fpr);
    }
    if (IsGroupFNRConstrained()) {
      constraint_proxy_object->ComputeGroupwiseFNR(score, group_fnr, num_data_, label_, weights_, group_,
                                                   group_values_);
      max_proxy_fnr = Constrained::findMaxValuePair<constraint_group_t, double>(group_fnr);
    }

    if (weights_ != nullptr) {
      throw std::logic_error("Weighted constraint gradients not implemented yet");
    }

    for (data_size_t i = 0; i < num_data_; ++i) {
      const auto group = group_[i];

      uint16_t number_of_groups = group_values_.size();
      uint16_t multipliers_base_index = 0;

      if (IsGroupFPRConstrained()) {
        if (label_[i] == 0) {
          double fpr_grad =
              (constraint_proxy_object->ComputeInstancewiseFPRGradient(score[i]) / group_label_negatives_.at(group));
          fpr_grad *= num_data_;

          if (group == max_proxy_fpr.first) {
            fpr_grad *= (number_of_groups - 1.);
            double lag_multipliers = 0;
            for (const auto& other_group : group_values_) {
              if (other_group != max_proxy_fpr.first)
                lag_multipliers += lagrangian_multipliers[multipliers_base_index + other_group];
            }
            gradients[i] += static_cast<score_t>(fpr_grad * lag_multipliers);
          } else {
            gradients[i] += static_cast<score_t>(-1. * fpr_grad *
                                                 lagrangian_multipliers[multipliers_base_index + group]);
          }
        }
        multipliers_base_index += number_of_groups;
      }

      if (IsGroupFNRConstrained()) {
        if (label_[i] == 1) {
          double fnr_grad =
              (constraint_proxy_object->ComputeInstancewiseFNRGradient(score[i]) / group_label_positives_.at(group));
          fnr_grad *= num_data_;

          if (group == max_proxy_fnr.first) {
            fnr_grad *= (number_of_groups - 1.0);
            double lag_multipliers = 0;
            for (const auto& other_group : group_values_) {
              if (other_group != max_proxy_fnr.first)
                lag_multipliers += lagrangian_multipliers[multipliers_base_index + other_group];
            }
            gradients[i] += static_cast<score_t>(fnr_grad * lag_multipliers);
          } else {
            gradients[i] += static_cast<score_t>(-1. * fnr_grad *
                                                 lagrangian_multipliers[multipliers_base_index + group]);
          }
        }
        multipliers_base_index += number_of_groups;
      }

      // Global Constraints
      if (IsGlobalFPRConstrained()) {
        if (label_[i] == 0) {
          double global_fpr_grad =
              (constraint_proxy_object->ComputeInstancewiseFPRGradient(score[i]) / total_label_negatives_);
          global_fpr_grad *= num_data_;
          gradients[i] += static_cast<score_t>(lagrangian_multipliers[multipliers_base_index] * global_fpr_grad);
        }
        multipliers_base_index += 1;
      }

      if (IsGlobalFNRConstrained()) {
        if (label_[i] == 1) {
          double global_fnr_grad =
              (constraint_proxy_object->ComputeInstancewiseFNRGradient(score[i]) / total_label_positives_);
          global_fnr_grad *= num_data_;
          gradients[i] += static_cast<score_t>(lagrangian_multipliers[multipliers_base_index] * global_fnr_grad);
        }
        multipliers_base_index += 1;
      }
    }
  }

  inline bool IsConstrained() const { return true; }

  // convert score to a probability
  inline void ConvertOutput(const double* input, double* output) const override {
    *output = 1.0f / (1.0f + std::exp(-(*input)));
  }

  inline bool IsGroupFPRConstrained() const {
    assert(group_constraint != UNSET);
    return group_constraint == FPR || group_constraint == FPR_AND_FNR;
  }

  inline bool IsGroupFNRConstrained() const {
    assert(group_constraint != UNSET);
    return group_constraint == FNR || group_constraint == FPR_AND_FNR;
  }

  inline bool IsGlobalFPRConstrained() const {
    assert(global_constraint != UNSET);
    return global_constraint == FPR || global_constraint == FPR_AND_FNR;
  }

  inline bool IsGlobalFNRConstrained() const {
    assert(global_constraint != UNSET);
    return global_constraint == FNR || global_constraint == FPR_AND_FNR;
  }

  int NumConstraints() const {
    int group_size = static_cast<int>(group_values_.size());
    int num_constraints = 0;

    if (IsGroupFPRConstrained())
      num_constraints += group_size;
    if (IsGroupFNRConstrained())
      num_constraints += group_size;
    if (IsGlobalFPRConstrained())
      num_constraints += 1;
    if (IsGlobalFNRConstrained())
      num_constraints += 1;

    return num_constraints;
  }

  void ComputeFPR(const double* score, double probabilities_threshold,
                  std::unordered_map<constraint_group_t, double>& group_fpr) const {
    std::unordered_map<int, int> false_positives;
    std::unordered_map<int, int> label_negatives;

    for (data_size_t i = 0; i < num_data_; ++i) {
      constraint_group_t group = group_[i];
      if (label_[i] == 0) {
        label_negatives[group] += 1;
        const double z = 1.0f / (1.0f + std::exp(-score[i]));
        if (z >= probabilities_threshold)
          false_positives[group] += 1;
      }
    }

    for (auto group_id : group_values_) {
      if (label_negatives[group_id] == 0)
        group_fpr[group_id] = 0;
      else
        group_fpr[group_id] = static_cast<double>(false_positives[group_id]) /
                              static_cast<double>(label_negatives[group_id]);
    }
  }

  double ComputeGlobalFPR(const double* score, double probabilities_threshold) const {
    int false_positives = 0, label_negatives = 0;
    for (data_size_t i = 0; i < num_data_; ++i) {
      if (label_[i] == 0) {
        label_negatives += 1;
        const double z = 1.0f / (1.0f + std::exp(-score[i]));
        if (z >= probabilities_threshold)
          false_positives += 1;
      }
    }
    return label_negatives > 0 ? static_cast<double>(false_positives) / static_cast<double>(label_negatives) : 0.0;
  }

  void ComputeFNR(const double* score, double probabilities_threshold,
                  std::unordered_map<constraint_group_t, double>& group_fnr) const {
    std::unordered_map<constraint_group_t, int> false_negatives;
    std::unordered_map<constraint_group_t, int> label_positives;

    for (data_size_t i = 0; i < num_data_; ++i) {
      constraint_group_t group = group_[i];
      if (label_[i] == 1) {
        label_positives[group] += 1;
        const double z = 1.0f / (1.0f + std::exp(-score[i]));
        if (z < probabilities_threshold)
          false_negatives[group] += 1;
      }
    }

    for (auto group_id : group_values_) {
      if (label_positives[group_id] == 0)
        group_fnr[group_id] = 0;
      else
        group_fnr[group_id] = static_cast<double>(false_negatives[group_id]) /
                              static_cast<double>(label_positives[group_id]);
    }
  }

  double ComputeGlobalFNR(const double* score, double probabilities_threshold) const {
    int false_negatives = 0, label_positives = 0;
    for (data_size_t i = 0; i < num_data_; ++i) {
      if (label_[i] == 1) {
        label_positives += 1;
        const double z = 1.0f / (1.0f + std::exp(-score[i]));
        if (z < probabilities_threshold)
          false_negatives += 1;
      }
    }
    return label_positives > 0 ? static_cast<double>(false_negatives) / static_cast<double>(label_positives) : 0.0;
  }

  void ComputeLabelCounts() {
    for (data_size_t i = 0; i < num_data_; ++i) {
      if (label_[i] == 1) {
        this->group_label_positives_[group_[i]] += 1;
        this->total_label_positives_ += 1;
      } else if (label_[i] == 0) {
        this->group_label_negatives_[group_[i]] += 1;
        this->total_label_negatives_ += 1;
      } else {
        throw std::runtime_error("invalid label type");
      }
    }
  }

 protected:
  static std::string ValidateProxyFunctionName(std::string func_name, bool allow_empty = false) {
    std::transform(func_name.begin(), func_name.end(), func_name.begin(), ::tolower);
    if (func_name == "bce" || func_name == "xentropy" || func_name == "entropy")
      func_name = "cross_entropy";

    if (!(func_name == "hinge" || func_name == "quadratic" || func_name == "cross_entropy" ||
          (allow_empty && func_name.empty()))) {
      throw std::invalid_argument("Got invalid proxy function: '" + func_name + "'");
    }
    return func_name;
  }

  /*! \brief Number of data points */
  data_size_t num_data_ = 0;
  /*! \brief Pointer for label */
  const label_t* label_ = nullptr;
  /*! \brief Weights for data */
  const label_t* weights_ = nullptr;

  /*! \brief Pointer for group */
  const constraint_group_t* group_ = nullptr;
  /*! \brief Unique group values */
  std::vector<constraint_group_t> group_values_;

  /*! \brief Label positives per group */
  std::unordered_map<constraint_group_t, int> group_label_positives_;
  /*! \brief Label Negatives per group */
  std::unordered_map<constraint_group_t, int> group_label_negatives_;

  /*! \brief Total number of Label Positives */
  int total_label_positives_ = 0;
  /*! \brief Total number of Label Negatives */
  int total_label_negatives_ = 0;

  /*! \brief Type of constraint */
  std::string constraint_type_str;
  /*! \brief Function to use as a proxy for the step-wise function in CONSTRAINTS. */
  std::string constraint_stepwise_proxy;
  /*! \brief Object to use as proxy for the step-wise function in CONSTRAINTS. */
  std::unique_ptr<ProxyLoss> constraint_proxy_object;
  /*! \brief Function to use as a proxy for the step-wise function in the OBJECTIVE. */
  std::string objective_stepwise_proxy;

  score_t score_threshold_ = 0.5;
  score_t fpr_threshold_ = 0.0;
  score_t fnr_threshold_ = 0.0;
  score_t proxy_margin_ = 1.0;

  std::string global_constraint_type_str;
  score_t global_target_fpr_ = 0.0;
  score_t global_target_fnr_ = 0.0;
  score_t global_score_threshold_ = 0.5;

  std::string debugging_output_dir_;

  constraint_type_t group_constraint = UNSET;
  constraint_type_t global_constraint = UNSET;
};
}  // namespace Constrained
}  // namespace LightGBM

#endif  // FAIRGBM_CONSTRAINED_OBJECTIVE_FUNCTION_H_
