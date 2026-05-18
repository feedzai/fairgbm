/**
 * Minimal LightGBM ObjectiveFunction stub for standalone FairGBM compilation.
 * Provides only the abstract interface that ConstrainedObjectiveFunction extends.
 */
#ifndef LIGHTGBM_OBJECTIVE_FUNCTION_H_
#define LIGHTGBM_OBJECTIVE_FUNCTION_H_

#include <LightGBM/meta.h>
#include <string>

namespace LightGBM {

/*!
 * \brief The interface for objective function (loss).
 * Minimal stub — only methods overridden by FairGBM are declared.
 */
class ObjectiveFunction {
 public:
  virtual ~ObjectiveFunction() = default;

  virtual void Init(const Metadata& metadata, data_size_t num_data) = 0;

  virtual void GetGradients(const double* score, score_t* gradients, score_t* hessians) const = 0;

  virtual void ConvertOutput(const double* input, double* output) const {
    *output = *input;
  }

  virtual const char* GetName() const = 0;

  virtual std::string ToString() const = 0;

  virtual double BoostFromScore(int class_id) const = 0;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_OBJECTIVE_FUNCTION_H_
