"""Unit tests for build configuration and package metadata."""

import configparser
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


class TestPyprojectToml:
    """Verify pyproject.toml declares correct metadata and dependencies."""

    @pytest.fixture(autouse=True)
    def _load_pyproject(self):
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        pyproject_path = ROOT / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml not found at project root"
        with open(pyproject_path, "rb") as f:
            self.config = tomllib.load(f)

    def test_lightgbm_declared_as_dependency(self):
        """Requirement 1.1, 10.4: lightgbm is a declared install dependency."""
        deps = self.config["project"]["dependencies"]
        lgbm_deps = [d for d in deps if d.startswith("lightgbm")]
        assert len(lgbm_deps) == 1
        assert ">=3.3.0" in lgbm_deps[0]
        assert "<5.0.0" in lgbm_deps[0]

    def test_numpy_declared_as_dependency(self):
        deps = self.config["project"]["dependencies"]
        assert any(d.startswith("numpy") for d in deps)

    def test_scipy_declared_as_dependency(self):
        deps = self.config["project"]["dependencies"]
        assert any(d.startswith("scipy") for d in deps)

    def test_scikit_learn_declared_as_dependency(self):
        deps = self.config["project"]["dependencies"]
        assert any(d.startswith("scikit-learn") for d in deps)

    def test_package_name(self):
        assert self.config["project"]["name"] == "fairgbm"

    def test_version_from_file(self):
        dynamic = self.config["project"]["dynamic"]
        assert "version" in dynamic
        version_file = ROOT / "VERSION.txt"
        assert version_file.exists()
        version = version_file.read_text().strip()
        assert version  # non-empty

    def test_build_requires_cmake(self):
        build_requires = self.config["build-system"]["requires"]
        assert any("cmake" in r for r in build_requires)


class TestSetupPy:
    """Verify setup.py has CMake build integration."""

    def test_setup_py_exists(self):
        assert (ROOT / "setup.py").exists()

    def test_setup_py_has_cmake_build_class(self):
        content = (ROOT / "setup.py").read_text()
        assert "CMakeBuild" in content or "BuildPyWithCMake" in content

    def test_setup_py_includes_shared_library_in_package_data(self):
        content = (ROOT / "setup.py").read_text()
        assert "lib_fairgbm" in content

    def test_setup_py_provides_error_message_on_cmake_failure(self):
        content = (ROOT / "setup.py").read_text()
        assert "CMake" in content
        assert "C++ compiler" in content or "build-essential" in content
