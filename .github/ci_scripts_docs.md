# CI Scripts Documentation

This document describes the shell scripts and utilities used by the continuous integration workflows in the FairGBM project. These scripts handle setup, testing, and workflow orchestration across different platforms and build configurations. They are primarily called by the GitHub Actions workflows documented in [ci_workflows_docs.md](./ci_workflows_docs.md).

The documentation below showcases each CI script's functionality.

## append_comment.sh
This script allows for updating a GitHub issue/PR comment by appending new text to an existing comment. It retrieves the original comment via the GitHub API, formats the new content, and patches the comment with the combined text.

```mermaid
graph TD
G[fairgbm/.ci/append_comment.sh, variables: comment_id=$1, body=$2]
    G --> H{if '$GITHUB_ACTIONS' is null}
    H --true--> J1{If the number of arguments is different to 2}
    H --false--> J2[echo 'Must be run inside GitHub Actions CI']
    J1 --> L1(Retrieve 'old_comment_body' by making a request to the GIthub API to obtain the comment from the '$GITHUB_API_URL/repos/microsoft/LightGBM/issues/comments/$comment_id' endpoint)
    J1 --true--> L2[echo 'echo 'Usage: $0 <COMMENT_ID> <BODY>'']

    G --> M(Status check on 'Body')
    M --> O[/Body/]
    L1 --> N[/old_comment_body/]
    N --> P('old_comment' and 'body' is formatted into data)
    O --> P
    P --> Q[/data/]
    Q --> R(curl -sL \
  -X PATCH \
  -H 'Accept: application/vnd.github.v3+json' \
  -H 'Authorization: token $SECRETS_WORKFLOW' \
  -d '$data' \
  '$GITHUB_API_URL/repos/microsoft/LightGBM/issues/comments/$comment_id')

    R -->S[end]
```

## fairgbm/.ci/get_workflow_status.py
```mermaid
graph TD
A[fairgbm/.ci/get_workflow_status.py]
    A --> C[/trigger code phrase that starts the pipelines/]
    C --> D(Obtains the most recent runs in the current PR via 'get_runs')
    D --> E[/recent runs/]
    E --> F(obtains status via 'get_status')
    F --> G[/run status/]
    G --> H{if 'status' is not 'in-progress'}
    H --true--> I1[break]
    H --> I2{if 'status' is failure'}
    I2 --true--> J[exit]
```

## fairgbm/.ci/install_opencl.ps1
Perform the OpenCL Installation

## fairgbm/.ci/lint_r_code.R
Linting utility for R code

## fairgbm/.ci/rerun_workflow.sh
Made to rerun a given workflow inside a pull request

```mermaid
graph TD
A[fairgbm/.ci/rerun_workflow.sh, variables: workflow_id=$1, pr_number=$2, pr_branch=$3]
    A --> B{If '$GITHUB_ACTIONS' is not empty}
    B --false--> C2[echo 'Must be run inside GitHub Actions CI']
    B --true--> C1{If the number of parameters is different than 3}
    C1 --true--> D2[echo 'Usage: $0 <WORKFLOW_ID> <PR_NUMBER> <PR_BRANCH>']
    C1 --false--> D1(obtain runs from the Github API)
    D1 --> E[/runs/]
    E --> F{If 'run''s length is above 0}
    F --> G1(Make a request to prompt the workflow via the Gihtub API)
    G1 --> H[end]
```

## fairgbm/.ci/run_rhub_solaris_checks.R
Run Solaris checks

## fairgbm/.ci/set_commit_status.sh
Set a status with a given name to the specified commit.

```mermaid
graph TD
A[fairgbm/.ci/set_commit_status.sh, variables: name=$1, status=$2, sha=$3]
    A --> B{If '$GITHUB_ACTIONS' is not empty}
    B --false--> C2[echo 'Must be run inside GitHub Actions CI']
    B --true--> C1{If the number of parameters is different than 3}
    C1 --true--> D2[echo 'Usage: $0 <NAME> <STATUS> <SHA>']
    C1 --false--> D1(status check with the 'status' variable)
    D1 --> E(Request to the Github API with the name and status variables)
    A --> E
    E --> F[/data/]
    F --> G(Request to the Github API URL to prompt the commit status update with the 'data' and the 'sha' variables)
    A --> G
    G --> H[end]
```

## fairgbm/.ci/setup.sh
Setup script for different platforms
```mermaid
graph TD
A[fairgbm/.ci/setup.sh]
  A --> B{If the OS is 'macos'}
  B --true--> C1{if the compiler is 'clang'}
  C1 --true--> D(brew install libomp)
  C1 --true--> E{if 'Azure' is true}
  E --true--> F1(sudo xcode-select -s /Applications/Xcode_10.3.app/Contents/Developer)
  F1 --> F2[exit]
  C1 --false--> G{if the task is not 'mpi'}
  G --true--> H(brew install gcc)
  B --true--> I{ig the task is 'mpi'}
  I --true--> J1(brew install open-mpi)
  B --true--> K1{If the task is 'swig'}
  K1 --true--> L1(brew install swig)
  B --true--> M(Download the miniforge conda script)

  B --false--> N{If 'IN_UBUNTU_LATEST_CONTAINER' is true}
  N --true--> O(install the necessary packages)
  N --true--> P{if the compiler is 'clang'}
  P --true--> Q(install clang)
  B --false--> R{if the task is 'mpi'}
  R --true--> S(install the necessary packages)
  B --false--> T{if the task is 'gpu'}
  T --true--> U1(install the relevant packages)
  T --true--> U2{if 'IN_UBUNTU_LATEST_CONTAINER' is true}
  U2 --true--> U3(install the relevant packages)
  B --false--> V{if the task is 'cuda' or 'cuda_exp'}

  D --> X{If task is different from 'r-package' or 'r-rchk'}
  H --> X
  J1 --> X
  L1 --> X
  M --> X
  O --> X
  Q --> X
  S --> X
  U1 --> X
  U3 --> X
  T --> X
  V2 --> X
  V4 --> X
  V5 --> X
  X4 --> X
  
  
  V --true--> V2(install the relevant packages)
  V --true--> V3{if the compiler is 'clang'}
  V3 --true--> V4(install the relevant packages)
  V --true--> V5(install the relevant packages)

  B --false--> W{if 'SETUP_CONDA' is not false}
  W --true--> W1(Download the relevant miniconda files)

  X --true--> X1{if 'SETUP_CONDA' is not false}
  X1 --true--> X3(install the conda package)
  W1 --true--> X4(run the relevant scripts)
```

## fairgbm/.ci/test.sh
Runs the test scripts for the different platforms and tasks


## fairgbm/.ci/test_r_package.sh, fairgbm/.ci/test_r_package_valgrind.sh, fairgbm/.ci/test_r_package_windows.ps1 and fairgbm/.ci/test_windows.ps1
These are test scripts for R in different platforms.

## fairgbm/.ci/trigger_dispatch_run.sh
Trigger manual workflow run by a dispatch event.

## Scripts Status and Maintenance Notes

### Legacy Scripts
The following scripts reference the original LightGBM repository and may be candidates for removal or adaptation:
- **append_comment.sh** - References `microsoft/LightGBM` in API calls; may not be actively used in FairGBM workflows
- **get_workflow_status.py** - May be legacy from LightGBM fork
- **rerun_workflow.sh** - Check if still used in current CI pipelines
- **run_rhub_solaris_checks.R** - Solaris support may not be required for FairGBM