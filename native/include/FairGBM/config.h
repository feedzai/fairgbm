/**
 * Copyright 2022 Feedzai
 *
 * FairGBM configuration structure.
 * When building inside LightGBM (Java build), this aliases LightGBM::Config.
 * When building standalone (Python build), this provides a self-contained struct.
 */

#ifndef FAIRGBM_CONFIG_H_
#define FAIRGBM_CONFIG_H_

// Detect if we're building inside full LightGBM (Java build context)
// The patched LightGBM config.h defines constraint_group_column and other FairGBM params
#if __has_include(<LightGBM/config.h>)
#include <LightGBM/config.h>
// Check if this is the patched config with FairGBM params
#ifdef LIGHTGBM_CONFIG_H_
namespace FairGBM {
  using Config = LightGBM::Config;
}  // namespace FairGBM
#define FAIRGBM_CONFIG_IS_LIGHTGBM_CONFIG 1
#endif
#endif

#ifndef FAIRGBM_CONFIG_IS_LIGHTGBM_CONFIG

#include <string>

namespace FairGBM {

/**
 * Configuration for FairGBM constrained objectives.
 * Contains all FairGBM-specific parameters that were previously added to
 * LightGBM's Config in the fork.
 */
struct Config {
  std::string constraint_type = "FPR,FNR";
  std::string constraint_stepwise_proxy = "cross_entropy";
  std::string objective_stepwise_proxy = "";
  double stepwise_proxy_margin = 1.0;
  double constraint_fpr_tolerance = 0.01;
  double constraint_fnr_tolerance = 0.01;
  double score_threshold = 0.5;
  std::string global_constraint_type = "";
  double global_target_fpr = 1.0;
  double global_target_fnr = 1.0;
  double global_score_threshold = 0.5;
  double multiplier_learning_rate = 0.1;
  std::string debugging_output_dir = ".";
  std::string init_lagrangian_multipliers = "";
  std::string constraint_group_column = "";
  bool deterministic = false;
};

}  // namespace FairGBM

#endif  // FAIRGBM_CONFIG_IS_LIGHTGBM_CONFIG

#endif  // FAIRGBM_CONFIG_H_
