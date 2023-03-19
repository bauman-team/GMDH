"""
A setuptools based setup module.
"""

import os
import subprocess
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):  # pylint: disable=too-few-public-methods, missing-class-docstring
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())

class CMakeBuild(build_ext):  # pylint: disable=missing-class-docstring
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        build_temp_str = str(os.path.join(Path(self.build_temp), ext.name))
        build_args = ['--config', 'Release', '--target', '_gmdh_core']

        subprocess.run(["cmake", "-DCMAKE_BUILD_TYPE=Release", "-S", ".", "-B", build_temp_str],
                       check=True)
        subprocess.run(["cmake", "--build", build_temp_str] + build_args, check=True)
        subprocess.run(["cp", str(os.path.join(build_temp, self.get_ext_filename(ext.name))),
                        str(os.path.join(extdir, "gmdh"))], check=True)

ext_modules = [
    CMakeExtension("_gmdh_core"),
]

with open("README.md", "r", encoding='utf8') as f:
    long_description = f.read()

version = {}
with open("gmdh/version.py", encoding='utf8') as f:
    exec(f.read(), version)  # pylint: disable=exec-used

project_urls = {
  'Documentation': 'https://bauman-team.github.io/GMDH',
  'GMDH book': 'https://bauman-team.github.io/GMDH_book',
  'Source Code': 'https://github.com/bauman-team/GMDH',
}

setup(
    name='gmdh',
    version=version['__version__'],
    author='Artem Babin',
    author_email='artem031201@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    url='https://github.com/bauman-team/GMDH',
    license='LICENSE.md',
    description="README.md",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls = project_urls,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
    install_requires=[
        "docstring_inheritance",
        "numpy",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires='>=3.6',
)
