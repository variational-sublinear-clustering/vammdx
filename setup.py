# Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

# install: pip install .
# develop: pip install --editable . // pip install -e .

import os
import toml
import pathlib
import sysconfig
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile

pyproject_text = pathlib.Path("pyproject.toml").read_text()
pyproject_data = toml.loads(pyproject_text)
build_type = pyproject_data["build-system"]["build-type"]


BUILD_TYPES = {
    "Release": ["-O3", "-DNDEBUG"],
    "Debug": ["-O0", "-g"],
    "RelWithDebInfo": ["-O2", "-g", "-DNDEBUG"],
    "MinSizeRel": ["-Os", "-DNDEBUG"],
}

include_dirs = [
    "vammdx/extern/vamm/vamm/extern/eigen",
    "vammdx/extern/vamm/vamm/cpp/include",
    "vammdx/cpp/include",
]

for lib in (
    "unordered",
    "assert",
    "container_hash",
    "config",
    "core",
    "predef",
    "throw_exception",
    "mp11",
    "describe",
    "static_assert",
):
    include_dirs += [f"vammdx/extern/vamm/vamm/extern/boost/{lib}/include"]

extra_compile_args = sysconfig.get_config_var("CFLAGS").split()
extra_compile_args += [
    "-Wall",
    "-Wextra",
    # "-Wshadow",
    "-pedantic",
    "-Wno-unknown-pragmas",
    "-march=native",
    "-DBOOST_ALLOW_DEPRECATED_HEADERS",
]
extra_compile_args += BUILD_TYPES.get(build_type, [])

define_macros = [("CLUSTERING_PRECISION", "double"), ("EIGEN_DONT_PARALLELIZE", None)]

ext_modules = [
    Pybind11Extension(
        "cppvammdx",
        [
            "vammdx/cpp/src/Bindings.cpp",
        ],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args + ["-fopenmp"],
        extra_link_args=["-lgomp"],
        define_macros=define_macros,
        language="c++",
        cxx_std=17,
    ),
]

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
with ParallelCompile(default=0):
    setup(
        name="vammdx",
        version="0.1",
        packages=find_packages(),
        zip_safe=False,
        ext_modules=ext_modules,
        install_requires=[
            "numpy",
            "scikit-learn",
            "pandas",
            "h5py",
            "scipy",
            "matplotlib",
            "imageio",
            f"vamm @ file://{PKG_DIR}/vammdx/extern/vamm",
            f"imageutils @ file://{PKG_DIR}/vammdx/extern/imageutils",
        ],
    )
