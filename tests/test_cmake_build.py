"""
Tests for CMake build configuration of the FairGBM native extension.

Validates:
- CMake configuration succeeds with bundled LightGBM header stubs
- CMake builds the shared library successfully
"""

import os
import subprocess
import tempfile

import pytest

NATIVE_DIR = os.path.join(os.path.dirname(__file__), "..", "native")
NATIVE_DIR = os.path.abspath(NATIVE_DIR)


def test_cmake_configures_successfully():
    """CMake configuration succeeds with bundled LightGBM header stubs."""
    with tempfile.TemporaryDirectory() as build_dir:
        result = subprocess.run(
            ["cmake", "-S", NATIVE_DIR, "-B", build_dir],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"CMake configure failed:\n{result.stderr}"
        assert "Using bundled LightGBM header stubs" in result.stdout


def test_cmake_builds_shared_library():
    """CMake builds lib_fairgbm.so successfully."""
    with tempfile.TemporaryDirectory() as build_dir:
        # Configure
        result = subprocess.run(
            ["cmake", "-S", NATIVE_DIR, "-B", build_dir],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"CMake configure failed:\n{result.stderr}"

        # Build
        result = subprocess.run(
            ["cmake", "--build", build_dir],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"CMake build failed:\n{result.stderr}"

        # Verify shared library exists
        lib_name = "lib_fairgbm.so"
        lib_path = os.path.join(build_dir, lib_name)
        assert os.path.isfile(lib_path), f"Expected {lib_name} not found in {build_dir}"


def test_cmake_openmp_detection():
    """CMake detects OpenMP when available."""
    with tempfile.TemporaryDirectory() as build_dir:
        result = subprocess.run(
            ["cmake", "-S", NATIVE_DIR, "-B", build_dir],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # OpenMP detection is optional — just verify it doesn't break the build
        if "OpenMP found" in result.stdout:
            assert "enabling parallel gradient computation" in result.stdout
