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

#include <FairGBM/fairgbm_c_api.h>
#include <FairGBM/constrained_objective_function.h>
#include <FairGBM/config.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Include objective implementations
#include "objective/constrained_xentropy_objective.hpp"
#include "objective/constrained_recall_objective.hpp"

using namespace LightGBM;
using namespace LightGBM::Constrained;

// ---------------------------------------------------------------------------
// Thread-local error string
// ---------------------------------------------------------------------------
static thread_local std::string last_error;

static void SetLastError(const char* msg) {
  last_error = msg;
}

static void SetLastError(const std::string& msg) {
  last_error = msg;
}

// ---------------------------------------------------------------------------
// Minimal Metadata adapter — wraps raw arrays to satisfy Init(const Metadata&, ...)
// ---------------------------------------------------------------------------
namespace {

/**
 * FairGBM's ConstrainedObjectiveFunction::Init expects a Metadata reference.
 * Since we don't have a full LightGBM Dataset, we provide a minimal adapter
 * that stores raw pointers and computes unique groups.
 *
 * NOTE: This class does NOT inherit from LightGBM::Metadata. Instead, the
 * C API bypasses Init() and directly sets the protected members of
 * ConstrainedObjectiveFunction via a friend-like pattern (see InitFromRaw below).
 */
struct RawMetadata {
  const label_t* labels = nullptr;
  const label_t* weights = nullptr;
  const constraint_group_t* constraint_groups = nullptr;
  std::vector<constraint_group_t> unique_groups;
  int num_data = 0;

  void Setup(const float* raw_labels, const int* raw_groups, const float* raw_weights, int n) {
    num_data = n;
    // labels: float* → label_t* (label_t is float on most builds)
    labels = reinterpret_cast<const label_t*>(raw_labels);
    weights = (raw_weights != nullptr) ? reinterpret_cast<const label_t*>(raw_weights) : nullptr;
    // constraint_groups: int* → constraint_group_t*
    // constraint_group_t is int in the non-compact build
    constraint_groups = reinterpret_cast<const constraint_group_t*>(raw_groups);

    // Compute unique groups
    std::set<constraint_group_t> group_set(raw_groups, raw_groups + n);
    unique_groups.assign(group_set.begin(), group_set.end());
  }
};

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Internal handle structure
// ---------------------------------------------------------------------------
struct FairGBMObjectiveState {
  std::unique_ptr<ConstrainedObjectiveFunction> objective;
  RawMetadata metadata;
  bool initialized = false;
};

// ---------------------------------------------------------------------------
// Config parsing helper
// ---------------------------------------------------------------------------
static FairGBM::Config ParseParamsString(const std::string& params_str) {
  FairGBM::Config config;
  // Parse key=value pairs separated by whitespace or newlines
  std::istringstream stream(params_str);
  std::string token;
  std::unordered_map<std::string, std::string> params_map;

  while (stream >> token) {
    auto eq_pos = token.find('=');
    if (eq_pos != std::string::npos) {
      std::string key = token.substr(0, eq_pos);
      std::string value = token.substr(eq_pos + 1);
      params_map[key] = value;
    }
  }

  // Map known FairGBM params to Config fields
  if (params_map.count("constraint_type"))
    config.constraint_type = params_map["constraint_type"];
  if (params_map.count("constraint_stepwise_proxy"))
    config.constraint_stepwise_proxy = params_map["constraint_stepwise_proxy"];
  if (params_map.count("objective_stepwise_proxy"))
    config.objective_stepwise_proxy = params_map["objective_stepwise_proxy"];
  if (params_map.count("stepwise_proxy_margin"))
    config.stepwise_proxy_margin = std::stod(params_map["stepwise_proxy_margin"]);
  if (params_map.count("constraint_fpr_tolerance"))
    config.constraint_fpr_tolerance = std::stod(params_map["constraint_fpr_tolerance"]);
  if (params_map.count("constraint_fnr_tolerance"))
    config.constraint_fnr_tolerance = std::stod(params_map["constraint_fnr_tolerance"]);
  if (params_map.count("score_threshold"))
    config.score_threshold = std::stod(params_map["score_threshold"]);
  if (params_map.count("global_constraint_type"))
    config.global_constraint_type = params_map["global_constraint_type"];
  if (params_map.count("global_target_fpr"))
    config.global_target_fpr = std::stod(params_map["global_target_fpr"]);
  if (params_map.count("global_target_fnr"))
    config.global_target_fnr = std::stod(params_map["global_target_fnr"]);
  if (params_map.count("global_score_threshold"))
    config.global_score_threshold = std::stod(params_map["global_score_threshold"]);
  if (params_map.count("multiplier_learning_rate"))
    config.multiplier_learning_rate = std::stod(params_map["multiplier_learning_rate"]);
  if (params_map.count("debugging_output_dir"))
    config.debugging_output_dir = params_map["debugging_output_dir"];
  if (params_map.count("deterministic"))
    config.deterministic = (params_map["deterministic"] == "true" || params_map["deterministic"] == "1");

  return config;
}

// ---------------------------------------------------------------------------
// Helper: directly initialize the objective's protected members from raw arrays
// (bypasses Metadata, which requires a full LightGBM Dataset)
// ---------------------------------------------------------------------------
static void InitObjectiveFromRaw(ConstrainedObjectiveFunction* obj, const RawMetadata& meta) {
  // Access protected members via the base class pointer.
  // We use a helper subclass trick to set them.
  // Since ConstrainedObjectiveFunction declares these as protected, and we
  // compile this in the same translation unit as the derived classes, we
  // use a reinterpret approach through a struct with identical layout.
  //
  // Actually, the cleanest approach: call Init with a real Metadata object
  // that we construct minimally. But Metadata's constructor requires a Dataset.
  //
  // Instead, we'll use a derived "accessor" class.
  struct Accessor : public ConstrainedObjectiveFunction {
    // Expose protected members for direct assignment
    using ConstrainedObjectiveFunction::num_data_;
    using ConstrainedObjectiveFunction::label_;
    using ConstrainedObjectiveFunction::weights_;
    using ConstrainedObjectiveFunction::group_;
    using ConstrainedObjectiveFunction::group_values_;
    using ConstrainedObjectiveFunction::total_label_positives_;
    using ConstrainedObjectiveFunction::total_label_negatives_;
    using ConstrainedObjectiveFunction::group_label_positives_;
    using ConstrainedObjectiveFunction::group_label_negatives_;
    using ConstrainedObjectiveFunction::ComputeLabelCounts;

    // Never instantiated — only used for member access
    double ComputePredictiveLoss(label_t, double) const override { return 0; }
    void GetGradients(const double*, score_t*, score_t*) const override {}
    const char* GetName() const override { return ""; }
    std::string ToString() const override { return ""; }
    double BoostFromScore(int) const override { return 0; }
  };

  auto* acc = static_cast<Accessor*>(obj);
  acc->num_data_ = static_cast<data_size_t>(meta.num_data);
  acc->label_ = meta.labels;
  acc->weights_ = meta.weights;
  acc->group_ = meta.constraint_groups;
  acc->group_values_ = meta.unique_groups;
  acc->total_label_positives_ = 0;
  acc->total_label_negatives_ = 0;
  acc->group_label_positives_.clear();
  acc->group_label_negatives_.clear();
  acc->ComputeLabelCounts();
}

// ---------------------------------------------------------------------------
// C API Implementation
// ---------------------------------------------------------------------------

const char* FairGBM_GetLastError(void) {
  return last_error.c_str();
}

int FairGBM_CreateConstrainedObjective(
    const char* objective_type,
    const char* params_str,
    ConstrainedObjectiveHandle* out) {
  try {
    if (objective_type == nullptr || out == nullptr) {
      SetLastError("Null argument passed to FairGBM_CreateConstrainedObjective");
      return -1;
    }

    FairGBM::Config config = ParseParamsString(params_str ? params_str : "");
    std::string obj_type(objective_type);

    auto* state = new FairGBMObjectiveState();

    if (obj_type == "constrained_cross_entropy") {
      state->objective.reset(new ConstrainedCrossEntropy(config));
    } else if (obj_type == "constrained_recall_objective") {
      state->objective.reset(new ConstrainedRecallObjective(config));
    } else {
      delete state;
      SetLastError("Unknown objective_type: " + obj_type +
                   ". Valid options: constrained_cross_entropy, constrained_recall_objective");
      return -1;
    }

    *out = static_cast<ConstrainedObjectiveHandle>(state);
    return 0;
  } catch (const std::exception& e) {
    SetLastError(e.what());
    return -1;
  }
}

int FairGBM_FreeConstrainedObjective(ConstrainedObjectiveHandle handle) {
  try {
    if (handle == nullptr) {
      return 0;  // freeing null is a no-op
    }
    auto* state = static_cast<FairGBMObjectiveState*>(handle);
    delete state;
    return 0;
  } catch (const std::exception& e) {
    SetLastError(e.what());
    return -1;
  }
}

int FairGBM_ObjectiveInit(
    ConstrainedObjectiveHandle handle,
    const float* labels,
    const int* constraint_groups,
    const float* weights,
    int num_data) {
  try {
    if (handle == nullptr) {
      SetLastError("Null handle in FairGBM_ObjectiveInit");
      return -1;
    }
    if (labels == nullptr || constraint_groups == nullptr || num_data <= 0) {
      SetLastError("Invalid arguments: labels and constraint_groups must be non-null, num_data > 0");
      return -1;
    }

    auto* state = static_cast<FairGBMObjectiveState*>(handle);
    state->metadata.Setup(labels, constraint_groups, weights, num_data);
    InitObjectiveFromRaw(state->objective.get(), state->metadata);
    state->initialized = true;
    return 0;
  } catch (const std::exception& e) {
    SetLastError(e.what());
    return -1;
  }
}

int FairGBM_GetGradients(
    ConstrainedObjectiveHandle handle,
    const double* scores,
    float* out_gradients,
    float* out_hessians) {
  try {
    if (handle == nullptr) {
      SetLastError("Null handle in FairGBM_GetGradients");
      return -1;
    }
    auto* state = static_cast<FairGBMObjectiveState*>(handle);
    if (!state->initialized) {
      SetLastError("Objective not initialized. Call FairGBM_ObjectiveInit first.");
      return -1;
    }

    state->objective->GetGradients(scores,
                                   reinterpret_cast<score_t*>(out_gradients),
                                   reinterpret_cast<score_t*>(out_hessians));
    return 0;
  } catch (const std::exception& e) {
    SetLastError(e.what());
    return -1;
  }
}

int FairGBM_GetConstraintGradients(
    ConstrainedObjectiveHandle handle,
    const double* lagrangian_multipliers,
    const double* scores,
    float* inout_gradients,
    float* inout_hessians) {
  try {
    if (handle == nullptr) {
      SetLastError("Null handle in FairGBM_GetConstraintGradients");
      return -1;
    }
    auto* state = static_cast<FairGBMObjectiveState*>(handle);
    if (!state->initialized) {
      SetLastError("Objective not initialized. Call FairGBM_ObjectiveInit first.");
      return -1;
    }

    state->objective->GetConstraintGradientsWRTModelOutput(
        lagrangian_multipliers, scores,
        reinterpret_cast<score_t*>(inout_gradients),
        reinterpret_cast<score_t*>(inout_hessians));
    return 0;
  } catch (const std::exception& e) {
    SetLastError(e.what());
    return -1;
  }
}

int FairGBM_GetLagrangianGradientsWRTMultipliers(
    ConstrainedObjectiveHandle handle,
    const double* scores,
    double* out_constraint_values,
    int* out_num_constraints) {
  try {
    if (handle == nullptr) {
      SetLastError("Null handle in FairGBM_GetLagrangianGradientsWRTMultipliers");
      return -1;
    }
    auto* state = static_cast<FairGBMObjectiveState*>(handle);
    if (!state->initialized) {
      SetLastError("Objective not initialized. Call FairGBM_ObjectiveInit first.");
      return -1;
    }

    std::vector<double> values = state->objective->GetLagrangianGradientsWRTMultipliers(scores);

    int n = static_cast<int>(values.size());
    if (out_num_constraints != nullptr) {
      *out_num_constraints = n;
    }
    if (out_constraint_values != nullptr) {
      std::memcpy(out_constraint_values, values.data(), n * sizeof(double));
    }
    return 0;
  } catch (const std::exception& e) {
    SetLastError(e.what());
    return -1;
  }
}

int FairGBM_GetNumConstraints(
    ConstrainedObjectiveHandle handle,
    int* out_num_constraints) {
  try {
    if (handle == nullptr) {
      SetLastError("Null handle in FairGBM_GetNumConstraints");
      return -1;
    }
    auto* state = static_cast<FairGBMObjectiveState*>(handle);

    // NumConstraints can be called before Init if config is set
    *out_num_constraints = state->objective->NumConstraints();
    return 0;
  } catch (const std::exception& e) {
    SetLastError(e.what());
    return -1;
  }
}

int FairGBM_BoostFromScore(
    ConstrainedObjectiveHandle handle,
    double* out_score) {
  try {
    if (handle == nullptr) {
      SetLastError("Null handle in FairGBM_BoostFromScore");
      return -1;
    }
    auto* state = static_cast<FairGBMObjectiveState*>(handle);
    if (!state->initialized) {
      SetLastError("Objective not initialized. Call FairGBM_ObjectiveInit first.");
      return -1;
    }

    *out_score = state->objective->BoostFromScore(0);
    return 0;
  } catch (const std::exception& e) {
    SetLastError(e.what());
    return -1;
  }
}
