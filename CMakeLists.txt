# FairGBM root CMakeLists.txt
#
# Strategy: At configure time, patch vendor/LightGBM with FairGBM extensions,
# then tell cmake to actually build from vendor/LightGBM by using a custom
# target that invokes cmake on the patched directory.

cmake_minimum_required(VERSION 3.14)
project(fairgbm_java_build LANGUAGES C CXX)

set(LIGHTGBM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/LightGBM")
set(FAIRGBM_NATIVE "${CMAKE_CURRENT_SOURCE_DIR}/native")

message(STATUS "[FairGBM] Patching vendor/LightGBM with FairGBM extensions...")

# --- Patch config ---
file(COPY_FILE "${FAIRGBM_NATIVE}/include/LightGBM/config.h" "${LIGHTGBM_DIR}/include/LightGBM/config.h")
file(COPY_FILE "${FAIRGBM_NATIVE}/include/LightGBM/config_auto.cpp" "${LIGHTGBM_DIR}/src/io/config_auto.cpp")

# --- Copy FairGBM headers ---
file(COPY "${FAIRGBM_NATIVE}/include/FairGBM" DESTINATION "${LIGHTGBM_DIR}/include/")
file(COPY_FILE "${FAIRGBM_NATIVE}/include/FairGBM/constrained_objective_function.h" "${LIGHTGBM_DIR}/include/LightGBM/constrained_objective_function.h")
file(COPY_FILE "${FAIRGBM_NATIVE}/include/FairGBM/utils/constrained.hpp" "${LIGHTGBM_DIR}/include/LightGBM/utils/constrained.hpp")
file(MAKE_DIRECTORY "${LIGHTGBM_DIR}/include/LightGBM/proxy_losses")
file(GLOB _ph "${FAIRGBM_NATIVE}/include/FairGBM/proxy_losses/*")
file(COPY ${_ph} DESTINATION "${LIGHTGBM_DIR}/include/LightGBM/proxy_losses/")

# --- Copy FairGBM sources ---
file(MAKE_DIRECTORY "${LIGHTGBM_DIR}/src/proxy_losses")
file(GLOB _ps "${FAIRGBM_NATIVE}/src/proxy_losses/*")
file(COPY ${_ps} DESTINATION "${LIGHTGBM_DIR}/src/proxy_losses/")
file(GLOB _os "${FAIRGBM_NATIVE}/src/objective/*")
file(COPY ${_os} DESTINATION "${LIGHTGBM_DIR}/src/objective/")

# --- Copy SWIG ---
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/swig/" DESTINATION "${LIGHTGBM_DIR}/swig/")

# --- Patch objective_function.cpp ---
file(READ "${LIGHTGBM_DIR}/src/objective/objective_function.cpp" _ofc)
string(FIND "${_ofc}" "constrained_xentropy" _idx)
if(_idx EQUAL -1)
    string(REPLACE "#include \"xentropy_objective.hpp\""
        "#include \"xentropy_objective.hpp\"\n#include \"constrained_xentropy_objective.hpp\"\n#include \"constrained_recall_objective.hpp\""
        _ofc "${_ofc}")
    string(REPLACE "} else if (type == std::string(\"cross_entropy_lambda\")) {\n    return new CrossEntropyLambda(config);"
        "} else if (type == std::string(\"constrained_cross_entropy\")) {\n    return new Constrained::ConstrainedCrossEntropy(config);\n  } else if (type == std::string(\"constrained_recall_objective\")) {\n    return new Constrained::ConstrainedRecallObjective(config);\n  } else if (type == std::string(\"cross_entropy_lambda\")) {\n    return new CrossEntropyLambda(config);"
        _ofc "${_ofc}")
    string(REPLACE "} else if (type == std::string(\"cross_entropy_lambda\")) {\n    return new CrossEntropyLambda(strs);"
        "} else if (type == std::string(\"constrained_cross_entropy\")) {\n    return new Constrained::ConstrainedCrossEntropy(strs);\n  } else if (type == std::string(\"constrained_recall_objective\")) {\n    return new Constrained::ConstrainedRecallObjective(strs);\n  } else if (type == std::string(\"cross_entropy_lambda\")) {\n    return new CrossEntropyLambda(strs);"
        _ofc "${_ofc}")
    file(WRITE "${LIGHTGBM_DIR}/src/objective/objective_function.cpp" "${_ofc}")
endif()

# --- Patch LightGBM CMakeLists to add proxy_losses ---
file(READ "${LIGHTGBM_DIR}/CMakeLists.txt" _lc)
string(FIND "${_lc}" "proxy_losses" _idx2)
if(_idx2 EQUAL -1)
    string(REPLACE "src/treelearner/*.cpp" "src/treelearner/*.cpp\n    src/proxy_losses/*.cpp" _lc "${_lc}")
    file(WRITE "${LIGHTGBM_DIR}/CMakeLists.txt" "${_lc}")
endif()

message(STATUS "[FairGBM] Patch complete.")
message(STATUS "[FairGBM] Configuring LightGBM build in vendor/LightGBM...")

# --- Configure and build LightGBM from vendor/ ---
# Use ExternalProject to build LightGBM as a separate cmake project
# This ensures PROJECT_SOURCE_DIR is correct inside LightGBM's CMakeLists
include(ExternalProject)
ExternalProject_Add(lightgbm_build
    SOURCE_DIR "${LIGHTGBM_DIR}"
    BINARY_DIR "${CMAKE_BINARY_DIR}/lgb"
    CMAKE_ARGS
        -DUSE_SWIG=${USE_SWIG}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    INSTALL_COMMAND ""
    BUILD_ALWAYS 1
)

# After build, copy artifacts to where the pipeline expects them
ExternalProject_Add_Step(lightgbm_build copy_artifacts
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LIGHTGBM_DIR}/lib_lightgbm.so" "${CMAKE_CURRENT_SOURCE_DIR}/lib_lightgbm.so"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LIGHTGBM_DIR}/lib_lightgbm_swig.so" "${CMAKE_CURRENT_SOURCE_DIR}/lib_lightgbm_swig.so"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_BINARY_DIR}/lgb/lightgbmlib.jar" "${CMAKE_BINARY_DIR}/lightgbmlib.jar"
    DEPENDEES build
    ALWAYS 1
)
