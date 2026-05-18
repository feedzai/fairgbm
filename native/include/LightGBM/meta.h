/**
 * Minimal LightGBM meta.h stub for standalone FairGBM compilation.
 * Provides only the type definitions used by FairGBM's constrained objectives.
 */
#ifndef LIGHTGBM_META_H_
#define LIGHTGBM_META_H_

#include <cstdint>
#include <limits>

namespace LightGBM {

/*! \brief Type of data size, it is better to use signed type */
typedef int32_t data_size_t;

/*! \brief Type of score (gradient/hessian) */
typedef float score_t;

/*! \brief Type of label */
typedef float label_t;

/*! \brief Small epsilon for numerical stability */
const double kEpsilon = 1e-15;

/*! \brief Minimal Metadata stub — not used in standalone FairGBM build */
class Metadata {
 public:
  virtual ~Metadata() = default;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_META_H_
