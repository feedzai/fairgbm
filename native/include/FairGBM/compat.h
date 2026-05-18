/**
 * Copyright 2022 Feedzai
 *
 * Compatibility header for building FairGBM against standard (upstream) LightGBM.
 * Defines types and stubs that exist in the FairGBM fork but not in upstream.
 */

#ifndef FAIRGBM_COMPAT_H_
#define FAIRGBM_COMPAT_H_

#include <LightGBM/meta.h>

namespace LightGBM {

// constraint_group_t is defined in the FairGBM fork's meta.h but not in upstream.
// We define it here as int (matching the non-compact fork definition).
typedef int constraint_group_t;

}  // namespace LightGBM

#endif  // FAIRGBM_COMPAT_H_
