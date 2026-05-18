cmake_minimum_required(VERSION 3.14)
project(fairgbm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- LightGBM headers ---
# FairGBM ships minimal LightGBM type stubs in native/include/LightGBM/
# so no external LightGBM headers are required at compile time.
# The stubs define only the types and interfaces used by FairGBM:
#   - data_size_t, score_t, label_t (meta.h)
#   - ObjectiveFunction abstract class (objective_function.h)
#   - Common::Join utility (utils/common.h)
message(STATUS "Using bundled LightGBM header stubs from ${CMAKE_SOURCE_DIR}/include/LightGBM/")

# --- FairGBM native extension ---
set(FAIRGBM_SOURCES
    src/proxy_losses/proxy_loss_factory.cpp
    src/fairgbm_c_api.cpp
)

add_library(lib_fairgbm SHARED ${FAIRGBM_SOURCES})

target_include_directories(lib_fairgbm PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(lib_fairgbm PROPERTIES
    PREFIX ""
    OUTPUT_NAME "lib_fairgbm"
)

# --- Optional OpenMP support ---
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(lib_fairgbm PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found, enabling parallel gradient computation")
endif()
