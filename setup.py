"""FairGBM setup with CMake native extension build."""

import os
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop


def _build_native_extension(output_dir: Path) -> None:
    """Run CMake configure + build, placing lib_fairgbm.so in output_dir."""
    native_dir = Path(__file__).parent / "native"
    build_dir = Path(__file__).parent / "build" / "cmake_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    cfg = "Release"
    cmake_args = [
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}",
        f"-DCMAKE_BUILD_TYPE={cfg}",
    ]

    build_args = ["--config", cfg]
    if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
        try:
            import multiprocessing

            build_args += [f"-j{multiprocessing.cpu_count()}"]
        except (ImportError, NotImplementedError):
            pass

    # Configure
    try:
        subprocess.run(
            ["cmake", str(native_dir), *cmake_args],
            cwd=build_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"CMake configuration failed.\n\n"
            f"stdout:\n{e.stdout}\n\nstderr:\n{e.stderr}\n\n"
            f"Ensure you have the following build dependencies installed:\n"
            f"  - CMake >= 3.14\n"
            f"  - A C++ compiler (g++, clang++, or MSVC)\n"
            f"  - lightgbm (pip install lightgbm)\n"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "CMake not found. Install CMake >= 3.14:\n"
            "  pip install cmake\n"
            "  or: apt-get install cmake  (Linux)\n"
            "  or: brew install cmake     (macOS)\n"
        )

    # Build
    try:
        subprocess.run(
            ["cmake", "--build", ".", *build_args],
            cwd=build_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"CMake build failed.\n\n"
            f"stdout:\n{e.stdout}\n\nstderr:\n{e.stderr}\n\n"
            f"Ensure you have a working C++ compiler installed:\n"
            f"  Linux:  apt-get install build-essential\n"
            f"  macOS:  xcode-select --install\n"
            f"  Windows: Install Visual Studio Build Tools\n"
        ) from e


class BuildPyWithCMake(build_py):
    """Custom build_py that compiles the native extension first."""

    def run(self):
        # Build native extension into the fairgbm package directory
        output_dir = (Path(__file__).parent / self.build_lib / "fairgbm").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        _build_native_extension(output_dir)
        super().run()


class DevelopWithCMake(develop):
    """Custom develop that compiles the native extension in-place."""

    def run(self):
        output_dir = Path(__file__).parent / "fairgbm"
        _build_native_extension(output_dir)
        super().run()


setup(
    cmdclass={
        "build_py": BuildPyWithCMake,
        "develop": DevelopWithCMake,
    },
    package_data={
        "fairgbm": ["lib_fairgbm.so", "lib_fairgbm.dylib", "lib_fairgbm.dll"],
    },
)
