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

#ifndef LIGHTGBM_CONSTRAINED_OBJECTIVE_FUNCTION_H_
#define LIGHTGBM_CONSTRAINED_OBJECTIVE_FUNCTION_H_

#include <LightGBM/config.h>
#include <LightGBM/dataset.h>
#include <LightGBM/meta.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/utils/constrained.hpp>
#include <LightGBM/proxy_losses/base.hpp>
#include <LightGBM/proxy_losses/hinge.hpp>
#include <LightGBM/proxy_losses/xentropy.hpp>
#include <LightGBM/proxy_losses/quadratic.hpp>
#include <LightGBM/proxy_losses/proxy_loss_factory.hpp>

#include <string>
#include <functional>

namespace LightGBM {
namespace Constrained {

class ConstrainedObjectiveFunction : public ObjectiveFunction
{
public:

    enum constraint_type_t { FPR, FNR, FPR_AND_FNR, NONE, UNSET };

    /*! \brief virtual destructor */
    ~ConstrainedObjectiveFunction() override = default;

    void SetUpFromConfig(const Config &config)
    {
      constraint_type_str = config.constraint_type;

      // Normalize constraint type
      std::transform(constraint_type_str.begin(), constraint_type_str.end(), constraint_type_str.begin(), ::toupper);
      if (constraint_type_str == "FNR,FPR")
        constraint_type_str = "FPR,FNR";

      fpr_threshold_ = (score_t) config.constraint_fpr_tolerance;
      fnr_threshold_ = (score_t) config.constraint_fnr_tolerance;
      score_threshold_ = (score_t) config.score_threshold;
      proxy_margin_ = (score_t) config.stepwise_proxy_margin;

      /** Global constraint parameters **/
      global_constraint_type_str = config.global_constraint_type;

      // Normalize global constraint type
      std::transform(global_constraint_type_str.begin(), global_constraint_type_str.end(), global_constraint_type_str.begin(), ::toupper);
      if (global_constraint_type_str == "FNR,FPR")
        global_constraint_type_str = "FPR,FNR";

      global_target_fpr_ = (score_t) config.global_target_fpr;
      global_target_fnr_ = (score_t) config.global_target_fnr;
      global_score_threshold_ = (score_t) config.global_score_threshold;

      // Function used as a PROXY for step-wise in the CONSTRAINTS
      constraint_stepwise_proxy = ValidateProxyFunctionName(config.constraint_stepwise_proxy, false);

      // Function used as a PROXY for the step-wise in the OBJECTIVE
      objective_stepwise_proxy = ValidateProxyFunctionName(config.objective_stepwise_proxy, true);

      // Debug configs
      debugging_output_dir_ = config.debugging_output_dir;

      // Construct ProxyLoss object for constraint functions
      constraint_proxy_object = ConstructProxyLoss(config);

      // Set type of GROUP constraints
      // (enums are much faster to compare than strings)
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
      * \brief Initialize
      * \param metadata Label data
      * \param num_data Number of data
      */
    void Init(const Metadata &metadata, data_size_t num_data) override
    {
      num_data_ = num_data;
      label_ = metadata.label();
      weights_ = metadata.weights();

      // Store Information about the group
      group_ = metadata.constraint_group();
      group_values_ = metadata.unique_constraint_groups();

      // Store Information about the labels
      total_label_positives_ = 0;
      total_label_negatives_ = 0;
      ComputeLabelCounts();

      CHECK_NOTNULL(label_);
      Common::CheckElementsIntervalClosed<label_t>(label_, 0.0f, 1.0f, num_data_, GetName());
      Log::Info("[%s:%s]: (objective) labels passed interval [0, 1] check", GetName(), __func__);

      if (weights_ != nullptr)
      {
        label_t minw;
        double sumw;
        Common::ObtainMinMaxSum(weights_, num_data_, &minw, static_cast<label_t *>(nullptr), &sumw);
        if (minw < 0.0f)
        {
          Log::Fatal("[%s]: at least one weight is negative", GetName());
        }
        if (sumw < DBL_MIN)
        {
          Log::Fatal("[%s]: sum of weights is zero", GetName());
        }
      }
    }

    /**
     * Template method for computing an instance's predictive loss value
     * from its predicted score (log-odds).
     *
     * @param label Instance label.
     * @param score Instance predicted score (log-odds);
     * @return The instance loss value.
     */
    virtual double ComputePredictiveLoss(label_t label, double score) const = 0;

    /*!
      * \brief Get functions w.r.t. to the lagrangian multipliers.
      * \brief This includes the evaluation of both the objective
      * \brief function (aka the loss) and also the (real) constraints.
      * \brief Therefore, the returned array will be of size.
      * \brief NumConstraints + 1 (plus one from the loss lagrang. multiplier).
      * \param score prediction score in this round.
      */
    virtual std::vector<double> GetLagrangianGradientsWRTMultipliers(const double *score) const
    {
      if (weights_ != nullptr)
        throw std::logic_error("not implemented yet");  // TODO: https://github.com/feedzai/fairgbm/issues/5

      std::vector<double> constraint_values;
      std::unordered_map<constraint_group_t, double> group_fpr, group_fnr;

      // NOTE! ** MULTIPLIERS ARE ORDERED! **
      //  - 1st: group-wise FPR constraints (one multiplier per group)
      //  - 2nd: group-wise FNR constraints (one multiplier per group)
      //  - 3rd: global FPR constraint      (a single multiplier)
      //  - 4th: global FNR constraint      (a single multiplier)

      // Multiplier corresponding to group-wise FPR constraints
      if (IsGroupFPRConstrained())
      {
        ComputeFPR(score, score_threshold_, group_fpr);
        double max_fpr = Constrained::findMaxValuePair<constraint_group_t, double>(group_fpr).second;

        // Assuming group_values_ is in ascending order
        for (const auto &group : group_values_)
        {
          double fpr_constraint_value = max_fpr - group_fpr[group] - fpr_threshold_;
          constraint_values.push_back(fpr_constraint_value);

#ifdef DEBUG
          Log::Debug(
            "DEBUG; true FPR constraint value: c = %.3f - %.3f = %.3f\n",
            max_fpr, group_fpr[group], fpr_constraint_value);
#endif
        }
      }

      // Multiplier corresponding to group-wise FNR constraints
      if (IsGroupFNRConstrained())
      {
        ComputeFNR(score, score_threshold_, group_fnr);
        double max_fnr = Constrained::findMaxValuePair<constraint_group_t, double>(group_fnr).second;

        // Assuming group_values_ is in ascending order
        for (const auto &group : group_values_)
        {
          double fnr_constraint_value = max_fnr - group_fnr[group] - fnr_threshold_;
          constraint_values.push_back(fnr_constraint_value);

#ifdef DEBUG
          Log::Debug(
            "DEBUG; true FNR constraint value: c = %.3f - %.3f = %.3f\n",
            max_fnr, group_fnr[group], fnr_constraint_value);
#endif
        }
      }

      // Next multiplier will correspond to the global FPR constraint
      if (IsGlobalFPRConstrained())
      {
        double global_fpr = ComputeGlobalFPR(score, global_score_threshold_);
        double global_fpr_constraint_value = global_fpr - global_target_fpr_;

        constraint_values.push_back(global_fpr_constraint_value);

#ifdef DEBUG
        Log::Debug(
          "DEBUG; true global FPR constraint value: c = %.3f - %.3f = %.3f\n",
          global_fpr, global_target_fpr_, global_fpr_constraint_value);
#endif
      }

      // Next multiplier will correspond to the global FNR constraint
      if (IsGlobalFNRConstrained())
      {
        double global_fnr = ComputeGlobalFNR(score, global_score_threshold_);
        double global_fnr_constraint_value = global_fnr - global_target_fnr_;

        constraint_values.push_back(global_fnr_constraint_value);

#ifdef DEBUG
        Log::Debug(
          "DEBUG; true global FNR constraint value: c = %.3f - %.3f = %.3f\n",
          global_fnr, global_target_fnr_, global_fnr_constraint_value);
#endif
      }

#ifdef DEBUG
      Constrained::write_values<double>(debugging_output_dir_, "constraint_values.dat", constraint_values);
#endif

      return constraint_values;
    }

    /*!
      * \brief Get gradients of the constraints w.r.t. to the scores (this will use proxy constraints!).
      * \param double Lagrangian multipliers in this round
      * \param score prediction score in this round
      * \gradients Output gradients
      * \hessians Output hessians
      */
    virtual void GetConstraintGradientsWRTModelOutput(const double *lagrangian_multipliers,
                                                      const double *score, score_t *gradients,
                                                      score_t * /* hessians */) const
    {

      std::unordered_map<constraint_group_t, double> group_fpr, group_fnr;
      std::pair<constraint_group_t, double> max_proxy_fpr, max_proxy_fnr;

      /** ---------------------------------------------------------------- *
       *                        FPR (Proxy) Constraint
       *  ---------------------------------------------------------------- *
       *  It corresponds to the result of differentiating the FPR proxy
       *  constraint w.r.t. the score of the ensemble.
       *
       *  FPR Proxy Constraints:
       *  lambda_group_i * [max(FPR_group_1, ..., FPR_group_j) - FPR_group_i]
       *
       *  ---------------------------------------------------------------- *
       *  To compute it, we need to:
       *  1. Compute FPR by group
       *  2. Determine the group with max(FPR)
       *  3. Compute derivative w.r.t. all groups except max(FPR)
       *  ---------------------------------------------------------------- *
       * */
      if (IsGroupFPRConstrained())
      {
        constraint_proxy_object->ComputeGroupwiseFPR(
                score, group_fpr, num_data_, label_, weights_, group_, group_values_);
        max_proxy_fpr = Constrained::findMaxValuePair<constraint_group_t, double>(group_fpr);
      }
      if (IsGroupFNRConstrained())
      {
        constraint_proxy_object->ComputeGroupwiseFNR(
                score, group_fnr, num_data_, label_, weights_, group_, group_values_);
        max_proxy_fnr = Constrained::findMaxValuePair<constraint_group_t, double>(group_fnr);
      }

      /** ---------------------------------------------------------------- *
       *                       GRADIENTS (per instance)                    *
       *  ---------------------------------------------------------------- */
      if (weights_ != nullptr)
      {
        throw std::logic_error("not implemented yet");  // TODO: https://github.com/feedzai/fairgbm/issues/5
      }

      // compute pointwise gradients and hessians with implied unit weights
//    #pragma omp parallel for schedule(static)       // TODO: https://github.com/feedzai/fairgbm/issues/6
      for (data_size_t i = 0; i < num_data_; ++i)
      {
        const auto group = group_[i];

        // Constraint index
        unsigned short number_of_groups = group_values_.size();
        unsigned short multipliers_base_index = 0;

        // -------------------------------------------------------------------
        // Skip FPR propagation if label positive, since LPs do not count for FPR constraints
        // -------------------------------------------------------------------
        // Grads of proxy constraints w.r.t. the scores:
        // (1) 0,    if label positive or score <= -margin (default margin=1)
        // (2) (m-1) / |LN_group_j| * (margin+score) * sum(lag multipliers except group j), if i belongs to group j whose FPR is maximal
        // (3) -lambda_k * (margin+score) / |LN_group_k| if the instance belongs to group k != j (where j has maximal FPR)
        // -------------------------------------------------------------------
        if (IsGroupFPRConstrained())
        {
          if (label_[i] == 0)
          {
            double fpr_constraints_gradient_wrt_pred = (
                    constraint_proxy_object->ComputeInstancewiseFPRGradient(score[i]) /
                    group_label_negatives_.at(group)
            );

            // Scale by dataset size, to avoid needing to scale the multiplier_learning_rate with the dataset size
            // See: https://github.com/feedzai/fairgbm/issues/7
            fpr_constraints_gradient_wrt_pred *= num_data_;

            // -------------------------------------------------------------------
            // Derivative (2) because instance belongs to group with maximal FPR
            // -------------------------------------------------------------------
            if (group == max_proxy_fpr.first)
            {
              // 2.1) Multiply by (m-1)
              fpr_constraints_gradient_wrt_pred *= (number_of_groups - 1.);

              // 2.2) Sum lagrangian multipliers (all except that of group with maximal FPR)
              double lag_multipliers = 0;
              for (const auto &other_group : group_values_)
              {
                if (other_group == max_proxy_fpr.first)
                  continue;
                else
                  lag_multipliers += lagrangian_multipliers[multipliers_base_index + other_group];  // NOTE: assumes group values start at zero (0)
              }

              gradients[i] += static_cast<score_t>(fpr_constraints_gradient_wrt_pred * lag_multipliers);
              // hessians[i] += ...
            }

            // ----------------------------------------------------------------------
            // Derivative (3) because instance belongs to group with non-maximal FPR
            // ----------------------------------------------------------------------
            else
            {
              gradients[i] += static_cast<score_t>(-1. * fpr_constraints_gradient_wrt_pred * lagrangian_multipliers[multipliers_base_index + group]);
              // hessians[i] += ...
            }
          }

          // Update index of multipliers to be used for next constraints
          multipliers_base_index += number_of_groups;
        }

        // Skip FNR propagation if label negative, since LNs do not count for FNR constraints
        if (IsGroupFNRConstrained())
        {
          if (label_[i] == 1)
          {
            double fnr_constraints_gradient_wrt_pred = (
                    constraint_proxy_object->ComputeInstancewiseFNRGradient(score[i]) /
                    group_label_positives_.at(group)
            );

            // Scale by dataset size, to avoid needing to scale the multiplier_learning_rate with the dataset size
            // See: https://github.com/feedzai/fairgbm/issues/7
            fnr_constraints_gradient_wrt_pred *= num_data_;

            // -------------------------------------------------------------------
            // Derivative (2) because instance belongs to group with max FNR
            // -------------------------------------------------------------------
            if (group == max_proxy_fnr.first)
            {
              // 2.1) Multiply by (m-1)
              fnr_constraints_gradient_wrt_pred *= (number_of_groups - 1.0);

              // 2.2) Sum lagrangian multipliers (all except that of group with maximal FNR)
              double lag_multipliers = 0;
              for (const auto &other_group : group_values_)
              {
                if (other_group == max_proxy_fnr.first)
                  continue;
                else
                  lag_multipliers += lagrangian_multipliers[multipliers_base_index + other_group];
              }

              gradients[i] += static_cast<score_t>(fnr_constraints_gradient_wrt_pred * lag_multipliers);
              // hessians[i] += ...
            }

            // ----------------------------------------------------------------------
            // Derivative (3) because instance belongs to group with non-maximal FNR
            // ----------------------------------------------------------------------
            else
            {
              gradients[i] += static_cast<score_t>(-1. * fnr_constraints_gradient_wrt_pred * lagrangian_multipliers[multipliers_base_index + group]);
              // hessians[i] += ...
            }
          }

          // Update index of multipliers to be used for next constraints
          multipliers_base_index += number_of_groups;
        }

        // ** Global Constraints **
        if (IsGlobalFPRConstrained())
        {
          if (label_[i] == 0)
          { // Condition for non-zero gradient
            double global_fpr_constraint_gradient_wrt_pred = (
                    constraint_proxy_object->ComputeInstancewiseFPRGradient(score[i]) /
                    total_label_negatives_
            );

            // Scale by dataset size, to avoid needing to scale the multiplier_learning_rate with the dataset size
            // See: https://github.com/feedzai/fairgbm/issues/7
            global_fpr_constraint_gradient_wrt_pred *= num_data_;

            // Update instance gradient and hessian
            gradients[i] += (score_t) (lagrangian_multipliers[multipliers_base_index] * global_fpr_constraint_gradient_wrt_pred);
            //          hessians[i] += ...
          }

          // Update index of multipliers to be used for next constraints
          multipliers_base_index += 1;
        }

        if (IsGlobalFNRConstrained())
        {
          if (label_[i] == 1)
          { // Condition for non-zero gradient
            double global_fnr_constraint_gradient_wrt_pred = (
                    constraint_proxy_object->ComputeInstancewiseFNRGradient(score[i]) /
                    total_label_positives_
            );

            // Scale by dataset size, to avoid needing to scale the multiplier_learning_rate with the dataset size
            // See: https://github.com/feedzai/fairgbm/issues/7
            global_fnr_constraint_gradient_wrt_pred *= num_data_;

            // Update instance gradient and hessian
            gradients[i] += (score_t)(lagrangian_multipliers[multipliers_base_index] *
                                      global_fnr_constraint_gradient_wrt_pred);
            //            hessians[i] += ...
          }

          // Update index of multipliers to be used for next constraints
          multipliers_base_index += 1;
        }
      }
    }

    inline bool IsConstrained() const override { return true; }

    // convert score to a probability
    inline void ConvertOutput(const double *input, double *output) const override
    {
      *output = 1.0f / (1.0f + std::exp(-(*input)));
    }

    inline bool IsGroupFPRConstrained() const
    {
      assert(group_constraint != UNSET);
      return group_constraint == FPR or group_constraint == FPR_AND_FNR;
    }

    inline bool IsGroupFNRConstrained() const
    {
      assert(group_constraint != UNSET);
      return group_constraint == FNR or group_constraint == FPR_AND_FNR;
    }

    inline bool IsGlobalFPRConstrained() const
    {
      assert(global_constraint != UNSET);
      return global_constraint == FPR or global_constraint == FPR_AND_FNR;
    }

    inline bool IsGlobalFNRConstrained() const
    {
      assert(global_constraint != UNSET);
      return global_constraint == FNR or global_constraint == FPR_AND_FNR;
    }

    int NumConstraints() const override
    {
      int group_size = (int) group_values_.size();
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

    /*!
      * \brief Computes group-wise false positive rate w.r.t. a given probabilities_threshold.
      * \param score prediction score in this round (logodds)
      * \param probabilities_threshold to consider for computing the FPR
      * \group_fpr Output the FPR per group
      */
    void ComputeFPR(const double *score, double probabilities_threshold, std::unordered_map<constraint_group_t, double> &group_fpr) const
    {
      std::unordered_map<int, int> false_positives;
      std::unordered_map<int, int> label_negatives;

      // #pragma omp parallel for schedule(static)        // TODO: https://github.com/feedzai/fairgbm/issues/6
      for (data_size_t i = 0; i < num_data_; ++i)
      {
        constraint_group_t group = group_[i];

        if (label_[i] == 0)
        {
          label_negatives[group] += 1;

          const double z = 1.0f / (1.0f + std::exp(-score[i]));
          if (z >= probabilities_threshold)
            false_positives[group] += 1;
        }
      }

      for (auto group_id : group_values_)
      {
        double fpr;
        if (label_negatives[group_id] == 0)
          fpr = 0;
        else
          fpr = ((double)false_positives[group_id]) / ((double)label_negatives[group_id]);

        group_fpr[group_id] = fpr;
      }
    }

    /**
     * Computes global False-Positive Rate according to the given threshold.
     * @param score
     * @param probabilities_threshold
     * @return the global FNR
     */
    double ComputeGlobalFPR(const double *score, double probabilities_threshold) const
    {
      int false_positives = 0, label_negatives = 0;

      // #pragma omp parallel for schedule(static)        // TODO: https://github.com/feedzai/fairgbm/issues/6
      for (data_size_t i = 0; i < num_data_; ++i)
      {
        if (label_[i] == 0)
        {
          label_negatives += 1;

          const double z = 1.0f / (1.0f + std::exp(-score[i]));
          if (z >= probabilities_threshold)
            false_positives += 1;
        }
      }

      return (double)false_positives / (double)label_negatives;
    }

    /*!
      * \brief Computes group-wise false negative rate w.r.t. a given probabilities_threshold.
      * \param score prediction score in this round (log-odds)
      * \param probabilities_threshold to consider for computing the FNR
      * \group_fnr Output the FNR per group
      */
    void ComputeFNR(const double *score, double probabilities_threshold, std::unordered_map<constraint_group_t, double> &group_fnr) const
    {
      std::unordered_map<constraint_group_t, int> false_negatives;
      std::unordered_map<constraint_group_t, int> label_positives;

      // #pragma omp parallel for schedule(static)        // TODO: https://github.com/feedzai/fairgbm/issues/6
      for (data_size_t i = 0; i < num_data_; ++i)
      {
        constraint_group_t group = group_[i];

        if (label_[i] == 1)
        {
          label_positives[group] += 1;

          const double z = 1.0f / (1.0f + std::exp(-score[i]));
          if (z < probabilities_threshold)
            false_negatives[group] += 1;
        }
      }

      for (auto group_id : group_values_)
      {
        double fnr;
        if (label_positives[group_id] == 0)
          fnr = 0;
        else
          fnr = ((double)false_negatives[group_id]) / ((double)label_positives[group_id]);
        group_fnr[group_id] = fnr;
      }
    };

    /**
     * Computes global False-Negative Rate according to the given threshold.
     * @param score
     * @param probabilities_threshold
     * @return the global FNR
     */
    double ComputeGlobalFNR(const double *score, double probabilities_threshold) const
    {
      int false_negatives = 0, label_positives = 0;

      // #pragma omp parallel for schedule(static)        // TODO: https://github.com/feedzai/fairgbm/issues/6
      for (data_size_t i = 0; i < num_data_; ++i)
      {
        if (label_[i] == 1)
        {
          label_positives += 1;

          const double z = 1.0f / (1.0f + std::exp(-score[i]));
          if (z < probabilities_threshold)
            false_negatives += 1;
        }
      }

      return (double)false_negatives / (double)label_positives;
    }

    /*!
      * \brief Get label positive and label negative counts.
      */
    void ComputeLabelCounts()
    {
      // #pragma omp parallel for schedule(static)        // TODO: https://github.com/feedzai/fairgbm/issues/6
      for (data_size_t i = 0; i < num_data_; ++i)
      {
        if (label_[i] == 1)
        {
          this->group_label_positives_[group_[i]] += 1;
          this->total_label_positives_ += 1;
        }

        else if (label_[i] == 0)
        {
          this->group_label_negatives_[group_[i]] += 1;
          this->total_label_negatives_ += 1;
        }

        else
          throw std::runtime_error("invalid label type");
      }
    };

protected:
    static std::string ValidateProxyFunctionName(std::string func_name, bool allow_empty = false)
    {
      std::transform(func_name.begin(), func_name.end(), func_name.begin(), ::tolower);
      if (func_name == "bce" or func_name == "xentropy" or func_name == "entropy")
        func_name = "cross_entropy";

      if (not(
              func_name == "hinge" or
              func_name == "quadratic" or
              func_name == "cross_entropy" or
              (allow_empty and func_name.empty())))
      {
        throw std::invalid_argument("Got invalid proxy function: '" + func_name + "'");
      }

      return func_name;
    }

    /*! \brief Number of data points */
    data_size_t num_data_;
    /*! \brief Pointer for label */
    const label_t *label_;
    /*! \brief Weights for data */
    const label_t *weights_;

    /*! \brief Pointer for group */
    const constraint_group_t *group_;
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

    /*! \brief Object to use as proxy for the ste-wise function in CONSTRAINTS. */
    std::unique_ptr<ProxyLoss> constraint_proxy_object;

    /*! \brief Function to use as a proxy for the step-wise function in the OBJECTIVE. */
    std::string objective_stepwise_proxy;

    /*! \brief Score threshold to compute confusion matrix (over predicted probabilities) */
    score_t score_threshold_ = 0.5;

    /*! \brief FPR threshold used in FPR constraints (small margin for constraint fulfillment) */
    score_t fpr_threshold_ = 0.0;

    /*! \brief FNR threshold used in FNR constraints (small margin for constraint fulfillment) */
    score_t fnr_threshold_ = 0.0;

    /*! \brief Margin threshold used in the Hinge approximation */
    score_t proxy_margin_ = 1.0;

    /*! \brief Type of global constraint */
    std::string global_constraint_type_str;

    /*! \brief Target value for the global FPR constraint */
    score_t global_target_fpr_;

    /*! \brief Target value for the global FNR constraint */
    score_t global_target_fnr_;

    /*! \brief Score threshold used for the global constraints */
    score_t global_score_threshold_ = 0.5;

    /*! \brief Where to save debug files to */
    std::string debugging_output_dir_;

    /*! \brief The type of group constraints in place */
    constraint_type_t group_constraint = UNSET;

    /*! \brief The type of global constraints in place */
    constraint_type_t global_constraint = UNSET;
};
}   // namespace Constrained
}

#endif  // LIGHTGBM_CONSTRAINED_OBJECTIVE_FUNCTION_H_
