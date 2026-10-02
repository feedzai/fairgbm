# Environment Setup Guide

This guide tells you how to prepare a workstation to build, install, and test FairGBM from the tracked repository. FairGBM has two parts: a C++ core (the boosting engine) and a Python package (the user-facing API). The tracked Python package compiles the C++ core for you during install. Read Section 1 before you install anything.

Get a working, fairness-capable FairGBM in about 15-25 minutes.

## 1. Prerequisites

### 1.1 Read this first: which Python package to build

The repository contains the tracked, functional Python package under `python-package/`. Build and install that one. This guide covers only that package.

| Item | Location | Tracked by git? | Use it? |
|---|---|---|---|
| **FairGBM Python package** | `python-package/` | Yes | **Yes.** Its `setup.py` compiles the C++ core and bundles the shared library. Its `fairgbm` module carries the full `constraint_group` fairness plumbing. |
| C++ core sources | `src/`, `include/`, `CMakeLists.txt` | Yes | Built automatically by the package install. |
| Experimental thin wrapper | `src/fairgbm/`, root `pyproject.toml` | **No (untracked)** | **No.** This is local, unfinished scaffolding. It rides on the stock `lightgbm` pip package and cannot run a constrained fit (`fit(constraint_group=...)` raises `TypeError`). Ignore it for setup. |

> **Note.** If your working tree contains `src/fairgbm/` and a root `pyproject.toml`, those are untracked local files. They do not affect the `python-package/` install, which uses its own `setup.py`.

### 1.2 Platform support

**This guide targets x86-64 (Intel/AMD64) only.** All build and test steps assume an x86-64 toolchain. The repository was written for x86-64; its pinned-era dependencies publish x86-64 wheels, and the shared library must match the Python interpreter architecture.

- The recommended platform is **x86-64 Linux**. This is the fully supported platform.
- **x86-64 macOS (Intel Macs)** is supported for source builds. See Section 2.2.
- **Apple Silicon (macOS/ARM)** must run the whole toolchain through **Rosetta 2 emulation**. Rosetta 2 runs x86-64 code on ARM hardware. Because the pinned dependencies are native x86-64 and install cleanly under Rosetta, this path works. See Section 2.1.
- **ARM Linux (aarch64)** has no emulation path in this guide. Use an x86-64 Linux machine, an x86-64 container (`docker --platform linux/amd64`), or the Docker image (`ndrcrz/fairgbm-miniconda`).
- **Windows** is supported for the C++ core only. Use WSL2 with an x86-64 Ubuntu image for the Python package.

### 1.3 Tools common to all platforms

- [Git](https://git-scm.com) with submodule support (v2.13+)
- A C++11-capable compiler (Section 2 lists the compiler per platform)
- **CMake 3.x** (version 3.5 to 3.31). Do not use CMake 4.x directly; see the CMake-4 note in Section 3.1.
- Python 3.8 or higher. The repository targets Python 3.8-3.9 (the CI versions). Newer Python works for the package if you pin `scikit-learn` as shown in Section 3.
- `pip` and `venv` (both ship with CPython)

### 1.4 Dependency management tools used in this repository

**This repository does not use `uv`, Poetry, Conda-lock, or any lockfile-based Python tool.** There is no `POETRY_VIRTUALENVS_CREATE` / `UV_PROJECT_ENVIRONMENT` setting to configure. You create a plain `venv` and install with `pip`.

| Layer | Tool | Where it is defined |
|---|---|---|
| Python package build | `setuptools` calling `cmake` as a subprocess | `python-package/setup.py` |
| Python virtual environment | Standard library `venv` | You create it |
| C++ build | `CMake` + a native compiler + `make` | `CMakeLists.txt` |
| C++ third-party libraries | Git submodules (Eigen, fmt, Boost.Compute, fast_double_parser) | `.gitmodules` |
| Runtime Python dependencies | `pip` | `python-package/setup.py` `install_requires` (`numpy`, `scipy`, `scikit-learn!=0.22.0`) |
| CI environment provisioning (optional) | `conda`/`miniforge`, via a shell script | `.ci/setup.sh` |

> **Dependency pinning warning.** The package's `install_requires` sets no upper bound on `scikit-learn`. The code uses an old scikit-learn API (`force_all_finite`) that was removed in scikit-learn 1.8. Pin `scikit-learn<1.6` at install time (Section 3.4), or a constrained fit fails with `TypeError: check_X_y() got an unexpected keyword argument 'force_all_finite'`.

## 2. Install a compiler and CMake (per platform)

### 2.1 macOS (Apple Silicon / ARM64) — toolchain under Rosetta 2

Install a second, x86-64 Homebrew under `/usr/local` and run every build command under `arch -x86_64`. Do not use your native ARM64 Homebrew (`/opt/homebrew`) for this build.

1. Install Rosetta 2 (skip if already present).
   ```bash
   softwareupdate --install-rosetta --agree-to-license
   ```
2. Install Xcode Command Line Tools. This gives you Apple Clang, which targets x86-64 under Rosetta.
   ```bash
   xcode-select --install
   ```
3. Install an x86-64 Homebrew under `/usr/local`. The `arch -x86_64` prefix forces the Intel install prefix.
   ```bash
   arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
4. Install the OpenMP runtime and an x86-64 Python through the x86-64 Homebrew. Apple Clang does not bundle OpenMP; you must add it separately. Do not install `cmake` from Homebrew here, because current Homebrew ships CMake 4.x; you will install CMake 3.x into the virtual environment instead (Section 3.3).
   ```bash
   arch -x86_64 /usr/local/bin/brew install libomp python@3.12
   ```
5. Open a Rosetta shell and keep every later command inside it, so the whole toolchain stays x86-64.
   ```bash
   arch -x86_64 zsh
   arch          # confirm it prints: i386
   uname -m      # confirm it prints: x86_64
   ```

### 2.2 macOS (Intel / x86-64)

Intel Macs are natively x86-64; no emulation is needed. If your Homebrew ships CMake 4.x, do not install CMake from Homebrew for this task; install CMake 3.x into the virtual environment (Section 3.3).

1. Install Xcode Command Line Tools.
   ```bash
   xcode-select --install
   ```
2. Install [Homebrew](https://brew.sh), then the OpenMP runtime.
   ```bash
   brew install libomp
   ```
3. Verify Apple Clang is version 8.1 or higher.
   ```bash
   clang --version
   ```

### 2.3 Linux (x86-64) — recommended platform

1. Install a compiler and build tools.
   ```bash
   sudo apt-get update
   sudo apt-get install --no-install-recommends -y build-essential
   ```
2. GCC users do not need an extra OpenMP package; `libgomp` ships with `build-essential`.
3. Install CMake. If your distribution's CMake is 3.x, use it. If it is 4.x, install CMake 3.x into the virtual environment instead (Section 3.3).
4. Verify the compiler version. GCC must be 4.8.2 or higher.
   ```bash
   gcc --version
   ```

### 2.4 Linux (ARM64 / aarch64) — not covered

Use an x86-64 Linux machine, an x86-64 container (`docker --platform linux/amd64`), or the Docker image (`ndrcrz/fairgbm-miniconda`).

### 2.5 Windows (x86-64 / Intel)

Use this platform only for the C++ core; use WSL2 (x86-64 Ubuntu) for the Python package. Install [Git for Windows](https://git-scm.com) and either Visual Studio 2015+ or CMake + VS Build Tools or CMake + MinGW-w64. MSVC must report version 1900 or higher.

## 3. Quick Start

On Apple Silicon, complete Section 2.1 and run every command below inside the Rosetta shell. On x86-64 Linux and Intel macOS, run the commands directly.

### 3.1 Clone with submodules

The C++ build needs vendored libraries (Eigen, fmt, Boost.Compute, fast_double_parser) that live in Git submodules. A plain clone leaves those directories empty.

```bash
git clone --recurse-submodules https://github.com/feedzai/fairgbm.git
cd fairgbm
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

> **CMake 4 incompatibility.** The tracked `CMakeLists.txt` declares `cmake_minimum_required(VERSION 3.2)` on its `APPLE` branch. CMake 4.x removed support for minimums below 3.5 and refuses to configure, and `setup.py` provides no way to pass a policy override. The reliable fix is to put a **CMake 3.x** first on `PATH`. Section 3.3 installs one into the virtual environment with `pip install "cmake<4"`.

### 3.2 Create a Python virtual environment

On x86-64 Linux and Intel macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Apple Silicon (inside the Rosetta shell), use the x86-64 Python from Section 2.1:

```bash
/usr/local/opt/python@3.12/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -c "import platform; print(platform.machine())"   # must print: x86_64
```

### 3.3 Install the build tooling

Install `setuptools`, `wheel`, and a **CMake 3.x** into the virtual environment. The CMake 3.x shadows any system CMake 4.x on `PATH`.

```bash
pip install --upgrade pip "setuptools>=70.1" wheel "cmake<4"
cmake --version    # must report 3.x (for example 3.31.x)
```

### 3.4 Install compatible runtime dependencies

Pin `scikit-learn` below 1.6 so the code's `force_all_finite` calls still work (Section 1.4).

```bash
pip install "scikit-learn<1.6"
```

On macOS, point CMake at the x86-64 OpenMP runtime so the compile finds it. This is an environment hint, not a file change.

```bash
export OpenMP_ROOT=/usr/local/opt/libomp     # macOS only (Intel brew prefix)
```

### 3.5 Build and install the FairGBM package

`setup.py` compiles the C++ core and bundles the shared library into the package. Use `--no-build-isolation` so the build uses the `cmake` you installed into the virtual environment. (With build isolation, the pip `cmake` wrapper script fails with `ModuleNotFoundError: No module named 'cmake'`.)

```bash
pip install --no-build-isolation ./python-package
```

This compiles the C++ core (one to three minutes) and installs `fairgbm` with its runtime dependencies (`numpy`, `scipy`, `scikit-learn`). The package is self-contained; it does not depend on the separate `lightgbm` pip package.

### 3.6 Install development and test dependencies (optional)

```bash
pip install pytest pandas
```

> **Running the bundled test suite.** `tests/python_test_requirements.txt` pins old versions (`scikit-learn~=1.0.2`, `pandas~=1.3.5`, `aequitas-lite~=0.43.2`) that have no wheels for Python 3.12. To run the bundled suite as written, use the CI-era Python (3.8 or 3.9) under Rosetta. For general development and use, Python 3.12 with `scikit-learn<1.6` is sufficient and is what the smoke test in Section 5 uses.

## 4. Manual Configuration

Follow this section to customize the build.

1. **Choose Python version by goal.** Use Python 3.8-3.9 to match CI and run the pinned test suite. Use a newer Python (with `scikit-learn<1.6`) for day-to-day development.
2. **Build options.** `setup.py` forwards a fixed set of flags to CMake. Useful ones: `--nomp` (disable OpenMP), `--gpu` (OpenCL), `--cuda`, `--mpi`. Example: `pip install --no-build-isolation ./python-package --config-settings=--global-option=--nomp` (consult `python-package/setup.py` for the current option names).
3. **OpenMP on macOS.** Keep `OpenMP_ROOT=/usr/local/opt/libomp` exported for the build. Apple Clang needs the Homebrew `libomp`; it does not bundle OpenMP.
4. **CMake selection.** Confirm `which cmake` points inside `.venv/bin` and reports 3.x before you build. A system CMake 4.x on `PATH` ahead of the virtual environment causes the configure step to fail.

## 5. Verify Installation

Run these checks from a directory outside the repository (so Python imports the installed package, not the source tree).

1. **Confirm the package imports and reports its architecture.**
   ```bash
   cd /tmp
   python -c "import fairgbm, platform; print('import OK;', 'arch', platform.machine())"
   ```
2. **Run a constrained-training smoke test.** This exercises the fairness path end to end.
   ```bash
   python - <<'PY'
   import numpy as np
   from fairgbm import FairGBMClassifier
   rng = np.random.default_rng(42)
   X = rng.normal(size=(500, 6))
   y = (rng.random(500) > 0.5).astype(int)
   s = (rng.random(500) > 0.5).astype(int)
   clf = FairGBMClassifier(constraint_type="FNR", n_estimators=25, random_state=42)
   clf.fit(X, y, constraint_group=s)
   p = clf.predict_proba(X)[:, -1]
   print("constrained fit OK; pred range [%.3f, %.3f]" % (p.min(), p.max()))
   PY
   ```
   Expected output: `import OK; arch x86_64` and `constrained fit OK; pred range [...]`.

## 6. Optional: full CI parity with conda

The CI pipeline (`.github/workflows/python_package.yml`) provisions the environment with `conda`/`miniforge` through `.ci/setup.sh`, then runs `.ci/test.sh`. You do not need conda for the Section 3 setup. Use this path only to reproduce CI exactly.

```bash
export OS_NAME="linux"     # or "macos"
export COMPILER="gcc"      # or "clang"
.ci/setup.sh               # installs system packages and Miniforge; read it first
export TASK="bdist"        # or: sdist, if-else, mpi
.ci/test.sh
```

## 7. Troubleshooting Quick Reference

| Symptom | Likely cause | Action |
|---|---|---|
| `ModuleNotFoundError: No module named 'cmake'` during build | pip build isolation hides the virtual-environment `cmake` wrapper | Add `--no-build-isolation` to `pip install ./python-package` (Section 3.5) |
| CMake error: `Compatibility with CMake < 3.5 has been removed` | System CMake is 4.x; the tracked `CMakeLists.txt` requests 3.2 | Install CMake 3.x into the virtual environment: `pip install "cmake<4"` (Section 3.3) |
| `TypeError: check_X_y() got an unexpected keyword argument 'force_all_finite'` | scikit-learn 1.8+ removed the argument the code uses | Install a compatible version: `pip install "scikit-learn<1.6"` (Section 3.4) |
| OpenMP / `omp.h` not found on macOS | Apple Clang has no bundled OpenMP, or CMake cannot locate `libomp` | `brew install libomp` and `export OpenMP_ROOT=/usr/local/opt/libomp` before building |
| `OSError: ... incompatible architecture (have 'arm64', need 'x86_64')` | A stale arm64 `lib_lightgbm.so` is being loaded in an x86-64 Python | Reinstall the package under Rosetta so the bundled library is x86-64; confirm the venv is x86-64 (`platform.machine()` → `x86_64`) |
| `arch: posix_spawnp: Bad CPU type in executable` | Rosetta 2 not installed | `softwareupdate --install-rosetta --agree-to-license` |
| `Cannot install under Rosetta 2 in ARM default prefix (/opt/homebrew)` | You used the native ARM Homebrew under emulation | Install the separate x86-64 Homebrew under `/usr/local` (Section 2.1 step 3) |
| `pip` installs latest `scikit-learn` and a fit then fails | `install_requires` has no scikit-learn upper bound | Pin `scikit-learn<1.6` (Section 3.4) |
| Empty `external_libs/*` directories | Submodules not initialized | `git submodule update --init --recursive` |
| Test suite will not install on Python 3.12 | `tests/python_test_requirements.txt` pins old, wheel-less versions | Use Python 3.8/3.9 for the pinned suite (Section 3.6) |
