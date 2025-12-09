# CI Workflows Documentation

This document describes the continuous integration (CI) workflows used in the FairGBM project. These workflows automate testing, validation, and publishing processes across different platforms and configurations. They can be triggered automatically on code changes (e.g. Pull Requests) or manually as needed.

The workflows utilize GitHub Actions and coordinate with the scripts documented in [ci_scripts_docs.md](./ci_scripts_docs.md).

The documentation below showcases each CI workflow's functionality.

## .github/workflows/cuda.yml
This script aims to test the project's CUDA support with the following different configurations:
- Source build - Python 3.7, GCC compiler, CUDA 11.2.2
- Pip install - Python 3.8, Clang compiler, CUDA 10.0
- Wheel install - Python 3.9, GCC compiler, CUDA 9.0

It triggers on pushing and creating pull requests to master.
Set environment variables:
  - github_actions: 'true'
  - os_name: linux
  - task: cuda
  - conda_env: test-env 

The steps are presented below:
### 1. Setup or update software on host machine
This step aims to setup the host machine with the necessary images and packages (Ubuntu image, Nvidia docker engine resources and docker)

### 2. Setup and run tests
This step aims to run the tests after the setup is done. This is achieved by setting up all the necessary variables and then running the docker image with any of the configuration combinations specified above.

## .github/workflows/linkchecker.yml
As the name suggests, this workflow is meant to check for broken links in the documentation. It is triggered with a cron schedule - once a day at 08:00am UTC. Otherwise, it can be triggered manually.

Set environment variables:
  - CONDA_ENV: test-env
  - GITHUB_ACTIONS: 'true'
  - OS_NAME: 'linux'
  - PYTHON_VERSION: 3.9
  - TASK: 'check-links'

The steps are disclosed below:

### 1. Checkout repository
This step checks out the relevant repo's last 5 commits without any submodules. The shallow clone with 5 commits (instead of just 1) allows for basic git operations and link checking against recent changes while keeping checkout time fast. This behaviour is caused by the `fetch-depth: 5` and the `submodules: false`, respectively.

### 2. Setup and run tests
Exports the relevant environment variables, and runs the [`.ci/setup.sh`](/.github/ci_scripts_docs.md#fairgbmcisetupsh) , which will install the necessary dependencies, and the [`.ci/test.sh`](/.github/ci_scripts_docs.md#fairgbmcitestsh), which will perform the link checking.

## .github/workflows/python_package.yml
This workflow is meant to test  the FairGBM package the following configuration combinations:
- sdist - Source distribution build (Python 3.9, Ubuntu)
- bdist - Binary distribution build (Python 3.8, Ubuntu)
- if-else - Tests conditional compilation paths (Python 3.8, Ubuntu)
- mpi + pip - MPI parallel learning via pip (Python 3.7, Ubuntu)
- mpi + wheel - MPI parallel learning via wheel (Python 3.7, Ubuntu)

This workflow is triggered on pushing and opening pull requests to the `master` and the `main-fairgbm`.

Set environment variables:
- CONDA_ENV: test-env
- GITHUB_ACTIONS: 'true'

This workflow's steps are the following:

### 1. Checkout repository
Checkout the actions repository, while only fetching the last 5 commits. In this instance however, the submodules are also fetched.

### 2. Setup and run tests
The relevant environment variables are set. Subsequently, the [`.ci/setup.sh`](/.github/ci_scripts_docs.md#fairgbmcisetupsh) and the [`.ci/test.sh`](/.github/ci_scripts_docs.md#fairgbmcitestsh) tasks are executed.

## .github/workflows/python_publish.yml
This workflow is meant to publish a python package into pypi. It is triggered on push with tags that match the `v*` expression.

The steps are explained below.

### 2. Set up Python
This step sets up python 3.7.

### 3. Install dependencies
As the name implies, this step install all the necesary dependencies.

### 4. Build package
This step [builds](https://pypi.org/project/build/) the designated python package.

### 5. Publish package
This step publishes the built package.

## .github/workflows/static_analysis.yml
This workflow performs static code analysis. It is triggered by pushing or creating pull requests to master.

The set environment variables are the following:
- COMPILER: 'gcc'
- CONDA_ENV: test-env
- GITHUB_ACTIONS: 'true'
- OS_NAME: 'linux'
- PYTHON_VERSION: 3.9

The steps are explained below. 


### 1. Checkout repository
The first step in this task is to checkout the relevant action's repository.

### 2. Setup and run tests
The following step runs the [`.ci/setup.sh`](/.github/ci_scripts_docs.md#fairgbmcisetupsh) and the [`.ci/test.sh`](/.github/ci_scripts_docs.md#fairgbmcitestsh) scripts. In this instance, this means to that the Python/C++ code is verified for style guidelines.

### r-check-docs
The next task is geared towards documentation and its steps are explained below.

### 1. Checkout repository
As explained above, this step checks out the relevant actions repository, its 5 latest commits and its submodules.

### 2. Install packages
This step installs the relevant R packages

### 3. Test documentation
This step checks for any changes made to the R documentation.