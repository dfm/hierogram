import os
import sys
import re

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup


if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

# Handle encoding
major, minor1, minor2, release, serial = sys.version_info
if major >= 3:
    def rd(filename):
        with open(filename, encoding="utf-8") as f:
            r = f.read()
        return r
else:
    def rd(filename):
        with open(filename) as f:
            r = f.read()
        return r

# Get the version number without actually importing.
vre = re.compile("__version__ = \"(.*?)\"")
m = rd(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "hierogram", "__init__.py"))
version = vre.findall(m)[0]

setup(
    name="hierogram",
    version=version,
    author="Dan Foreman-Mackey",
    author_email="dan@dfm.io",
    packages=["hierogram"],
    url="http://dan.iel.fm/hierogram/",
    license="MIT",
    description="Making histograms of noisy observations",
    long_description=rd("README.rst"),
    package_data={"": ["LICENSE"]},
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
    ],
    classifiers=[
        # "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
