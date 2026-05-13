/**
 * Minimal LightGBM Common utilities stub for standalone FairGBM compilation.
 * Provides only the functions used by FairGBM (Common::Join).
 */
#ifndef LIGHTGBM_UTILS_COMMON_H_
#define LIGHTGBM_UTILS_COMMON_H_

#include <sstream>
#include <string>
#include <vector>

namespace LightGBM {
namespace Common {

/*!
 * \brief Join a vector of values into a delimited string.
 */
template <typename T, typename Allocator>
inline std::string Join(const std::vector<T, Allocator>& values, const char* delimiter) {
  if (values.empty()) return "";
  std::ostringstream oss;
  oss << values[0];
  for (size_t i = 1; i < values.size(); ++i) {
    oss << delimiter << values[i];
  }
  return oss.str();
}

template <typename T>
inline std::string Join(const std::vector<T>& values, const char* delimiter) {
  if (values.empty()) return "";
  std::ostringstream oss;
  oss << values[0];
  for (size_t i = 1; i < values.size(); ++i) {
    oss << delimiter << values[i];
  }
  return oss.str();
}

}  // namespace Common
}  // namespace LightGBM

#endif  // LIGHTGBM_UTILS_COMMON_H_
