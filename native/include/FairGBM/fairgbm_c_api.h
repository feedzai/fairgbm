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

#ifndef FAIRGBM_C_API_H_
#define FAIRGBM_C_API_H_

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define FAIRGBM_C_EXPORT __declspec(dllexport)
#else
#define FAIRGBM_C_EXPORT __attribute__((visibility("default")))
#endif

/**
 * Opaque handle to a constrained objective instance.
 */
typedef void* ConstrainedObjectiveHandle;

/**
 * Get the last error message.
 * \return The last error message (thread-local).
 */
FAIRGBM_C_EXPORT const char* FairGBM_GetLastError(void);

/**
 * Create a constrained objective function.
 * \param objective_type One of "constrained_cross_entropy" or "constrained_recall_objective"
 * \param params_str Key=value pairs separated by whitespace (same format as LightGBM config)
 * \param[out] out Handle to the created objective
 * \return 0 on success, -1 on error
 */
FAIRGBM_C_EXPORT int FairGBM_CreateConstrainedObjective(
    const char* objective_type,
    const char* params_str,
    ConstrainedObjectiveHandle* out);

/**
 * Free a constrained objective handle.
 * \param handle Handle to free
 * \return 0 on success, -1 on error
 */
FAIRGBM_C_EXPORT int FairGBM_FreeConstrainedObjective(
    ConstrainedObjectiveHandle handle);

/**
 * Initialize the objective with training data.
 * \param handle Objective handle
 * \param labels Binary labels array (float, values in {0, 1}), length num_data
 * \param constraint_groups Integer constraint group array, length num_data
 * \param weights Per-instance weights (nullable), length num_data
 * \param num_data Number of training instances
 * \return 0 on success, -1 on error
 */
FAIRGBM_C_EXPORT int FairGBM_ObjectiveInit(
    ConstrainedObjectiveHandle handle,
    const float* labels,
    const int* constraint_groups,
    const float* weights,
    int num_data);

/**
 * Compute predictive loss gradients (without constraint terms).
 * \param handle Objective handle
 * \param scores Raw model scores, length num_data
 * \param[out] out_gradients Output gradients array, length num_data
 * \param[out] out_hessians Output hessians array, length num_data
 * \return 0 on success, -1 on error
 */
FAIRGBM_C_EXPORT int FairGBM_GetGradients(
    ConstrainedObjectiveHandle handle,
    const double* scores,
    float* out_gradients,
    float* out_hessians);

/**
 * Compute constraint gradient contributions and add to existing gradients/hessians.
 * \param handle Objective handle
 * \param lagrangian_multipliers Current multiplier values, length num_constraints
 * \param scores Raw model scores, length num_data
 * \param[in,out] inout_gradients Gradients to modify in-place, length num_data
 * \param[in,out] inout_hessians Hessians to modify in-place, length num_data
 * \return 0 on success, -1 on error
 */
FAIRGBM_C_EXPORT int FairGBM_GetConstraintGradients(
    ConstrainedObjectiveHandle handle,
    const double* lagrangian_multipliers,
    const double* scores,
    float* inout_gradients,
    float* inout_hessians);

/**
 * Compute Lagrangian gradients w.r.t. multipliers (constraint violation values).
 * \param handle Objective handle
 * \param scores Raw model scores, length num_data
 * \param[out] out_constraint_values Output constraint values, length num_constraints
 * \param[out] out_num_constraints Number of constraints written
 * \return 0 on success, -1 on error
 */
FAIRGBM_C_EXPORT int FairGBM_GetLagrangianGradientsWRTMultipliers(
    ConstrainedObjectiveHandle handle,
    const double* scores,
    double* out_constraint_values,
    int* out_num_constraints);

/**
 * Get the number of constraints for the current configuration.
 * \param handle Objective handle
 * \param[out] out_num_constraints Number of constraints
 * \return 0 on success, -1 on error
 */
FAIRGBM_C_EXPORT int FairGBM_GetNumConstraints(
    ConstrainedObjectiveHandle handle,
    int* out_num_constraints);

/**
 * Get the initial score (boost from average).
 * \param handle Objective handle
 * \param[out] out_score The initial score value
 * \return 0 on success, -1 on error
 */
FAIRGBM_C_EXPORT int FairGBM_BoostFromScore(
    ConstrainedObjectiveHandle handle,
    double* out_score);

#ifdef __cplusplus
}
#endif

#endif  /* FAIRGBM_C_API_H_ */
