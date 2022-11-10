import os
import subprocess
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:

        build_args = ['--config', 'Release', '--target', '_gmdh_core']

        subprocess.run(["cmake", "-DCMAKE_BUILD_TYPE=Release", "-S", ".", "-B", "build/Release"], check=True)
        subprocess.run(["cmake", "--build", "build/Release"] + build_args, check=True)

setup(
    name='gmdh',
    version='0.4.1',
    author='Artem Babin, Michail Baryyshnikov',
    author_email='artem031201@gmail.com',
    packages=find_packages(),
    url='https://github.com/bauman-team/GMDH',
    license='LICENSE.md',
    description='Gmdh algorithms',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
    ],
    install_requires=[
        "docstring_inheritance",
        "numpy",
    ],
    ext_modules=[CMakeExtension("_gmdh_core")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires='>=3.6',
)
