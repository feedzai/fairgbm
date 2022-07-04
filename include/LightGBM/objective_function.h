/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_OBJECTIVE_FUNCTION_H_
#define LIGHTGBM_OBJECTIVE_FUNCTION_H_

#include <LightGBM/config.h>
#include <LightGBM/dataset.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/constrained.hpp>

#include <string>
#include <functional>

namespace LightGBM {
/*!
* \brief The interface of Objective Function.
*/
class ObjectiveFunction {
 public:
  /*! \brief virtual destructor */
  virtual ~ObjectiveFunction() {}

  /*!
  * \brief Initialize
  * \param metadata Label data
  * \param num_data Number of data
  */
  virtual void Init(const Metadata& metadata, data_size_t num_data) = 0;

  /*!
  * \brief calculating first order derivative of loss function
  * \param score prediction score in this round
  * \gradients Output gradients
  * \hessians Output hessians
  */
  virtual void GetGradients(const double* score,
    score_t* gradients, score_t* hessians) const = 0;

  virtual const char* GetName() const = 0;

  virtual bool IsConstantHessian() const { return false; }

  virtual bool IsConstrained() const { return false; }

  virtual bool IsRenewTreeOutput() const { return false; }

  virtual double RenewTreeOutput(double ori_output, std::function<double(const label_t*, int)>,
                                 const data_size_t*,
                                 const data_size_t*,
                                 data_size_t) const { return ori_output; }

  virtual double BoostFromScore(int /*class_id*/) const { return 0.0; }

  virtual bool ClassNeedTrain(int /*class_id*/) const { return true; }

  virtual bool SkipEmptyClass() const { return false; }

  virtual int NumModelPerIteration() const { return 1; }

  virtual int NumPredictOneRow() const { return 1; }

  virtual int NumConstraints() const { return 0; }

  /*! \brief The prediction should be accurate or not. True will disable early stopping for prediction. */
  virtual bool NeedAccuratePrediction() const { return true; }

  /*! \brief Return the number of positive samples. Return 0 if no binary classification tasks.*/
  virtual data_size_t NumPositiveData() const { return 0; }

  virtual void ConvertOutput(const double* input, double* output) const {
    output[0] = input[0];
  }

  virtual std::string ToString() const = 0;

  ObjectiveFunction() = default;
  /*! \brief Disable copy */
  ObjectiveFunction& operator=(const ObjectiveFunction&) = delete;
  /*! \brief Disable copy */
  ObjectiveFunction(const ObjectiveFunction&) = delete;

  /*!
  * \brief Create object of objective function
  * \param type Specific type of objective function
  * \param config Config for objective function
  */
  LIGHTGBM_EXPORT static ObjectiveFunction* CreateObjectiveFunction(const std::string& type,
    const Config& config);

  /*!
  * \brief Load objective function from string object
  */
  LIGHTGBM_EXPORT static ObjectiveFunction* CreateObjectiveFunction(const std::string& str);
};

class ConstrainedObjectiveFunction : public ObjectiveFunction
{
public:
  /*! \brief virtual destructor */
  virtual ~ConstrainedObjectiveFunction() {}

  void SetUpFromConfig(const Config &config)
  {
    constraint_type = config.constraint_type;

    // Normalize constraint type
    std::transform(constraint_type.begin(), constraint_type.end(), constraint_type.begin(), ::toupper);
    if (constraint_type == "FNR,FPR")
      constraint_type = "FPR,FNR";

    fpr_threshold_ = (score_t)config.constraint_fpr_threshold;
    fnr_threshold_ = (score_t)config.constraint_fnr_threshold;
    score_threshold_ = (score_t)config.score_threshold;
    proxy_margin_ = (score_t)config.stepwise_proxy_margin;

    /** Global constraint parameters **/
    global_constraint_type = config.global_constraint_type;

    // Normalize global constraint type
    std::transform(global_constraint_type.begin(), global_constraint_type.end(), global_constraint_type.begin(), ::toupper);
    if (global_constraint_type == "FNR,FPR")
      global_constraint_type = "FPR,FNR";

    global_target_fpr_ = (score_t)config.global_target_fpr;
    global_target_fnr_ = (score_t)config.global_target_fnr;
    global_score_threshold_ = (score_t)config.global_score_threshold;

    // Function used as a PROXY for step-wise in the CONSTRAINTS
    constraint_stepwise_proxy = ValidateProxyFunctionName(config.constraint_stepwise_proxy, false);

    // Function used as a PROXY for the step-wise in the OBJECTIVE
    objective_stepwise_proxy = ValidateProxyFunctionName(config.objective_stepwise_proxy, true);

    // Debug configs
    debugging_output_dir_ = config.debugging_output_dir;
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
    if (IsFPRConstrained())
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
    if (IsFNRConstrained())
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

    // Helper constant for BCE-based proxies
    double xent_horizontal_shift = log(exp(proxy_margin_) - 1); // here, proxy_margin_ is the VERTICAL margin

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
    if (IsFPRConstrained())
    {
      if (constraint_stepwise_proxy == "hinge")
        ComputeHingeFPR(score, group_fpr);
      else if (constraint_stepwise_proxy == "quadratic")
        ComputeQuadraticLossFPR(score, group_fpr);
      else if (constraint_stepwise_proxy == "cross_entropy")
        ComputeXEntropyLossFPR(score, group_fpr);
      else
        throw std::invalid_argument("constraint_stepwise_proxy=" + constraint_stepwise_proxy + " not implemented!");

      max_proxy_fpr = Constrained::findMaxValuePair<constraint_group_t, double>(group_fpr);
    }
    if (IsFNRConstrained())
    {
      if (constraint_stepwise_proxy == "hinge")
        ComputeHingeLossFNR(score, group_fnr);
      else if (constraint_stepwise_proxy == "quadratic")
        ComputeQuadraticLossFNR(score, group_fnr);
      else if (constraint_stepwise_proxy == "cross_entropy")
        ComputeXEntropyLossFNR(score, group_fnr);
      else
        throw std::invalid_argument("constraint_stepwise_proxy=" + constraint_stepwise_proxy + " not implemented!");

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
    #pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < num_data_; ++i)
    {
      const auto group = group_[i];

      // Constraint index
      u_short number_of_groups = group_values_.size();
      u_short multipliers_base_index = 0;

      // -------------------------------------------------------------------
      // Skip FPR propagation if label positive, since LPs do not count for FPR constraints
      // -------------------------------------------------------------------
      // Grads of proxy constraints w.r.t. the scores:
      // (1) 0,    if label positive or score <= -margin (default margin=1)
      // (2) (m-1) / |LN_group_j| * (margin+score) * sum(lag multipliers except group j), if i belongs to group j whose FPR is maximal
      // (3) -lambda_k * (margin+score) / |LN_group_k| if the instance belongs to group k != j (where j has maximal FPR)
      // -------------------------------------------------------------------
      if (IsFPRConstrained())
      {
        if (label_[i] == 0)
        {
          const int group_ln = group_label_negatives_.at(group);

          double fpr_constraints_gradient_wrt_pred;
          // TODO: https://github.com/feedzai/fairgbm/issues/7

          // Derivative for hinge-based proxy FPR
          if (constraint_stepwise_proxy == "hinge")
            fpr_constraints_gradient_wrt_pred = score[i] <= -proxy_margin_ ? 0. : 1. / group_ln;

          // Derivative for BCE-based proxy FPR
          else if (constraint_stepwise_proxy == "cross_entropy") {
            fpr_constraints_gradient_wrt_pred = (Constrained::sigmoid(score[i] + xent_horizontal_shift)) / group_ln;
//            fpr_constraints_gradient_wrt_pred = (Constrained::sigmoid(score[i]) - label_[i]) / group_ln;   // without margin
          }

          // Loss-function implicitly defined as having a hinge-based derivative (quadratic loss)
          else if (constraint_stepwise_proxy == "quadratic") {
            fpr_constraints_gradient_wrt_pred = std::max(0., score[i] + proxy_margin_) / group_ln;
          }

          else
            throw std::invalid_argument("constraint_stepwise_proxy=" + constraint_stepwise_proxy + " not implemented!");

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
                lag_multipliers += lagrangian_multipliers[multipliers_base_index + other_group];
            }

            gradients[i] += static_cast<score_t>(fpr_constraints_gradient_wrt_pred * lag_multipliers);
            // hessians[i] += ...
          }
          else
          {
            // ----------------------------------------------------------------------
            // Derivative (3) because instance belongs to group with non-maximal FPR
            // ----------------------------------------------------------------------
            gradients[i] += static_cast<score_t>(-1. * fpr_constraints_gradient_wrt_pred * lagrangian_multipliers[multipliers_base_index + group]);
            // hessians[i] += ...
          }
        }

        // Update index of multipliers to be used for next constraints
        multipliers_base_index += number_of_groups;
      }

      // Skip FNR propagation if label negative, since LNs do not count for FNR constraints
      if (IsFNRConstrained())
      {
        if (label_[i] == 1)
        {
          const int group_lp = group_label_positives_.at(group);

          double fnr_constraints_gradient_wrt_pred;

          // Derivative for hinge-based proxy FNR
          if (constraint_stepwise_proxy == "hinge")
            fnr_constraints_gradient_wrt_pred = score[i] >= proxy_margin_ ? 0. : -1. / group_lp;

          // Derivative for BCE-based proxy FNR
          else if (constraint_stepwise_proxy == "cross_entropy") {
            fnr_constraints_gradient_wrt_pred = (Constrained::sigmoid(score[i] - xent_horizontal_shift) - 1) / group_lp;
//            fnr_constraints_gradient_wrt_pred = (Constrained::sigmoid(score[i]) - label_[i]) / group_lp;   // without margin
          }

          // Loss-function implicitly defined as having a hinge-based derivative (quadratic loss)
          else if (constraint_stepwise_proxy == "quadratic") {
            fnr_constraints_gradient_wrt_pred = std::min(0., score[i] - proxy_margin_) / group_lp;
          }

          else
            throw std::invalid_argument("constraint_stepwise_proxy=" + constraint_stepwise_proxy + " not implemented!");

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
          else
          {
            // ----------------------------------------------------------------------
            // Derivative (3) because instance belongs to group with non-maximal FNR
            // ----------------------------------------------------------------------
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
          double global_fpr_constraint_gradient_wrt_pred;
          // Gradient for hinge proxy FPR
          if (constraint_stepwise_proxy == "hinge") {
            global_fpr_constraint_gradient_wrt_pred = score[i] >= -proxy_margin_ ? 1. / total_label_negatives_ : 0.;
          }

          // Gradient for BCE proxy FPR
          else if (constraint_stepwise_proxy == "cross_entropy") {
            global_fpr_constraint_gradient_wrt_pred = (Constrained::sigmoid(score[i] + xent_horizontal_shift)) / total_label_negatives_;
//            global_fpr_constraint_gradient_wrt_pred = (Constrained::sigmoid(score[i]) - label_[i]) / total_label_negatives_;   // without margin
          }

          // Hinge-based gradient (for quadratic proxy FPR)
          else if (constraint_stepwise_proxy == "quadratic") {
            global_fpr_constraint_gradient_wrt_pred = std::max(0., score[i] + proxy_margin_) / total_label_negatives_;
          }

          else
            throw std::invalid_argument("constraint_stepwise_proxy=" + constraint_stepwise_proxy + " not implemented!");

          // Update instance gradient and hessian
          gradients[i] += (score_t)(lagrangian_multipliers[multipliers_base_index] * global_fpr_constraint_gradient_wrt_pred);
          //          hessians[i] += ...
        }

        // Update index of multipliers to be used for next constraints
        multipliers_base_index += 1;
      }

      if (IsGlobalFNRConstrained())
      {
        if (label_[i] == 1)
        { // Condition for non-zero gradient
          double global_fnr_constraint_gradient_wrt_pred;

          // Gradient for hinge proxy FNR
          if (constraint_stepwise_proxy == "hinge") {
            global_fnr_constraint_gradient_wrt_pred = score[i] >= proxy_margin_ ? 0. : -1. / total_label_positives_;
          }

          // Gradient for BCE proxy FNR
          else if (constraint_stepwise_proxy == "cross_entropy") {
            global_fnr_constraint_gradient_wrt_pred = (Constrained::sigmoid(score[i] - xent_horizontal_shift) - 1) / total_label_positives_;
//            global_fnr_constraint_gradient_wrt_pred = (Constrained::sigmoid(score[i]) - label_[i]) / total_label_positives_;   // without margin
          }

          // Hinge-based gradient (for quadratic proxy FNR)
          else if (constraint_stepwise_proxy == "quadratic") {
            global_fnr_constraint_gradient_wrt_pred = std::min(0., score[i] - proxy_margin_) / total_label_positives_;
          }

          else {
            throw std::invalid_argument("constraint_stepwise_proxy=" + constraint_stepwise_proxy + " not implemented!");
          }

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

  bool IsConstrained() const override { return true; }

  // convert score to a probability
  void ConvertOutput(const double *input, double *output) const override
  {
    *output = 1.0f / (1.0f + std::exp(-(*input)));
  }

  bool IsFPRConstrained() const
  {
    return (constraint_type == "FPR" || constraint_type == "FPR,FNR");
    // NOTE: Order of constraints in config file doesn't matter, it's sorted beforehand
  }

  bool IsFNRConstrained() const
  {
    return (constraint_type == "FNR" || constraint_type == "FPR,FNR");
  }

  bool IsGlobalFPRConstrained() const
  {
    return (global_constraint_type == "FPR" || global_constraint_type == "FPR,FNR");
  }

  bool IsGlobalFNRConstrained() const
  {
    return (global_constraint_type == "FNR" || global_constraint_type == "FPR,FNR");
  }

  int NumConstraints() const override
  {
    int group_size = (int)group_values_.size();
    int num_constraints = 0;

    if (IsFPRConstrained())
      num_constraints += group_size;
    if (IsFNRConstrained())
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
    * \brief Get hinge-proxy false positive rate w.r.t. a given margin
    * \param array of scores -> prediction score in this round
    * \param margin to consider for computing the Hinge approximation of FPR
    * \group_fpr Output the proxy FPR per group
    */
  void ComputeHingeFPR(const double *score, std::unordered_map<constraint_group_t, double> &group_fpr) const
  {
    std::unordered_map<constraint_group_t, double> false_positives; // map of group index to the respective hinge-proxy FPs
    std::unordered_map<constraint_group_t, int> label_negatives;    // map of group index to the respective number of LNs

    // #pragma omp parallel for schedule(static)        // TODO: https://github.com/feedzai/fairgbm/issues/6
    for (data_size_t i = 0; i < num_data_; ++i)
    {
      constraint_group_t group = group_[i];

      // HingeFPR uses only label negatives
      if (label_[i] == 0)
      {
        label_negatives[group] += 1;

        // proxy_margin_ is the line intercept value
        const double hinge_score = proxy_margin_ + score[i];
        false_positives[group] += std::max(0.0, hinge_score);
      }
    }

    for (auto group_id : group_values_)
    {
      double fpr;
      if (label_negatives[group_id] == 0)
        fpr = 0;
      else
        fpr = false_positives[group_id] / label_negatives[group_id];

      group_fpr[group_id] = fpr;
    }
  }

  /**
   * Compute quadratic-proxy FPR (with a given margin).
   *
   * Proxy FPR: (1/2) * (H_i + margin)^2 * I[H_i > -margin and y_i == 0]
   *
   * @param score array of scores
   * @param group_fpr hash-map of group to proxy-FPR
   */
  void ComputeQuadraticLossFPR(const double *score, std::unordered_map<constraint_group_t, double> &group_fpr) const
  {
    std::unordered_map<constraint_group_t, double> false_positives; // map of group index to the respective proxy FPs
    std::unordered_map<constraint_group_t, int> label_negatives;    // map of group index to the respective number of LNs

    // #pragma omp parallel for schedule(static)        // TODO: https://github.com/feedzai/fairgbm/issues/6
    for (data_size_t i = 0; i < num_data_; ++i)
    {
      constraint_group_t group = group_[i];

      // FPR uses only label NEGATIVES
      if (label_[i] == 0 and score[i] > -proxy_margin_)
      { // Conditions for non-zero proxy-FPR value
        label_negatives[group] += 1;

        // proxy_margin_ corresponds to the symmetric of the function's zero point; f(-proxy_margin_)=0
        const double quadratic_score = (1. / 2.) * std::pow(score[i] + proxy_margin_, 2);
        assert(quadratic_score >= 0.);
        false_positives[group] += quadratic_score;
      }
    }

    for (auto group_id : group_values_)
    {
      double fpr;
      if (label_negatives[group_id] == 0)
        fpr = 0;
      else
        fpr = false_positives[group_id] / label_negatives[group_id];

      group_fpr[group_id] = fpr;
    }
  }

  /**
   * Compute cross-entropy-proxy FPR.
   * Function:
   *      l(a) = log(1 + exp( a + log(exp(b) - 1) )),      where b = proxy_margin_ = l(0)
   *
   * @param score array of scores
   * @param group_fpr hash-map of group to proxy-FPR
   */
  void ComputeXEntropyLossFPR(const double *score, std::unordered_map<constraint_group_t, double> &group_fpr) const
  {
    std::unordered_map<constraint_group_t, double> false_positives; // map of group index to the respective proxy FPs
    std::unordered_map<constraint_group_t, int> label_negatives;    // map of group index to the respective number of LNs
    double xent_horizontal_shift = log(exp(proxy_margin_) - 1);

    // #pragma omp parallel for schedule(static)        // TODO: https://github.com/feedzai/fairgbm/issues/6
    for (data_size_t i = 0; i < num_data_; ++i)
    {
      constraint_group_t group = group_[i];

      // FPR uses only label NEGATIVES
      if (label_[i] == 0)
      {
        label_negatives[group] += 1;

        // proxy_margin_ corresponds to the vertical margin at x=0; l(0) = proxy_margin_
        const double xent_score = log(1 + exp(score[i] + xent_horizontal_shift));
        assert(xent_score >= 0.);
        false_positives[group] += xent_score;
      }
    }

    for (auto group_id : group_values_)
    {
      double fpr;
      if (label_negatives[group_id] == 0)
        fpr = 0;
      else
        fpr = false_positives[group_id] / label_negatives[group_id];

      group_fpr[group_id] = fpr;
    }
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
    * \brief Get hinge-proxy FNR w.r.t. a given margin.
    * \param score prediction score in this round
    * \param margin to consider for computing the FNR
    * \group_fnr Output the proxy FNR per group
    */
  void ComputeHingeLossFNR(const double *score, std::unordered_map<constraint_group_t, double> &group_fnr) const
  {
    std::unordered_map<constraint_group_t, double> false_negatives; // map of group index to the respective hinge-proxy FNs
    std::unordered_map<constraint_group_t, int> label_positives;

    // #pragma omp parallel for schedule(static)        // TODO: https://github.com/feedzai/fairgbm/issues/6
    for (data_size_t i = 0; i < num_data_; ++i)
    {
      constraint_group_t group = group_[i];

      if (label_[i] == 1)
      {
        label_positives[group] += 1;

        const double hinge_score = proxy_margin_ - score[i];
        false_negatives[group] += std::max(0.0, hinge_score);
      }
    }

    for (auto group_id : group_values_)
    {
      double fnr;
      if (label_positives[group_id] == 0)
        fnr = 0;
      else
        fnr = false_negatives[group_id] / label_positives[group_id];
      group_fnr[group_id] = fnr;
    }
  };

  /**
   * Compute quadratic-proxy FNR (with a given margin).
   *
   * Proxy FNR: (1/2) * (H_i - margin)^2 * I[H_i < margin and y_i == 1]
   *
   * @param score array of scores
   * @param group_fnr hash-map of group to proxy-FNR
   */
  void ComputeQuadraticLossFNR(const double *score, std::unordered_map<constraint_group_t, double> &group_fnr) const
  {
    std::unordered_map<constraint_group_t, double> false_negatives; // map of group index to the respective proxy FPs
    std::unordered_map<constraint_group_t, int> label_positives;    // map of group index to the respective number of LNs

    // #pragma omp parallel for schedule(static)        // TODO: https://github.com/feedzai/fairgbm/issues/6
    for (data_size_t i = 0; i < num_data_; ++i)
    {
      constraint_group_t group = group_[i];

      // FNR uses only label POSITIVES
      if (label_[i] == 1 and score[i] < proxy_margin_)
      { // Conditions for non-zero proxy-FNR value
        label_positives[group] += 1;

        // proxy_margin_ corresponds to the function's zero point; f(proxy_margin_)=0
        const double quadratic_score = (1. / 2.) * std::pow(score[i] - proxy_margin_, 2);
        assert(quadratic_score >= 0.);
        false_negatives[group] += quadratic_score;
      }
    }

    for (auto group_id : group_values_)
    {
      double fnr;
      if (label_positives[group_id] == 0)
        fnr = 0;
      else
        fnr = false_negatives[group_id] / label_positives[group_id];

      group_fnr[group_id] = fnr;
    }
  }

  /**
   * Compute cross-entropy-proxy FNR.
   * Function:
   *      l(a) = log(1 + exp( -a + log(exp(b) - 1) )),        where b = proxy_margin_ = l(0)
   *
   * @param score array of scores
   * @param group_fnr hash-map of group to proxy-FNR
   */
  void ComputeXEntropyLossFNR(const double *score, std::unordered_map<constraint_group_t, double> &group_fnr) const
  {
    std::unordered_map<constraint_group_t, double> false_negatives; // map of group index to the respective proxy FPs
    std::unordered_map<constraint_group_t, int> label_positives;    // map of group index to the respective number of LNs
    double xent_horizontal_shift = log(exp(proxy_margin_) - 1);

    // #pragma omp parallel for schedule(static)        // TODO: https://github.com/feedzai/fairgbm/issues/6
    for (data_size_t i = 0; i < num_data_; ++i)
    {
      constraint_group_t group = group_[i];

      // FNR uses only label POSITIVES
      if (label_[i] == 1)
      {
        label_positives[group] += 1;

        // proxy_margin_ corresponds to the vertical margin at x=0; l(0) = proxy_margin_
        const double xent_score = log(1 + exp(xent_horizontal_shift - score[i]));
        assert(xent_score >= 0.);
        false_negatives[group] += xent_score;
      }
    }

    for (auto group_id : group_values_)
    {
      double fnr;
      if (label_positives[group_id] == 0)
        fnr = 0;
      else
        fnr = false_negatives[group_id] / label_positives[group_id];

      group_fnr[group_id] = fnr;
    }
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
  std::string constraint_type;

  /*! \brief Function to use as a proxy for the step-wise function in CONSTRAINTS. */
  std::string constraint_stepwise_proxy;

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
  std::string global_constraint_type;

  /*! \brief Target value for the global FPR constraint */
  score_t global_target_fpr_;

  /*! \brief Target value for the global FNR constraint */
  score_t global_target_fnr_;

  /*! \brief Score threshold used for the global constraints */
  score_t global_score_threshold_ = 0.5;

  /*! \brief Where to save debug files to */
  std::string debugging_output_dir_;
};
} // namespace LightGBM

#endif   // LightGBM_OBJECTIVE_FUNCTION_H_
