"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""
import os
import sys

from setuptools import setup, find_packages

# pylint: disable=redefined-builtin

here = os.path.abspath(os.path.dirname(__file__))  # pylint: disable=invalid-name

with open(os.path.join(here, "README.rst"), encoding="utf-8") as fid:
    long_description = fid.read()  # pylint: disable=invalid-name

with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as fid:
    install_requires = [line for line in fid.read().splitlines() if line.strip()]

setup(
    name="ski-leu",
    # Don't forget to update the version in __init__.py and CHANGELOG.rst!
    version="0.0.1",
    description="Ski through a dangerous forest.",
    long_description=long_description,
    url="https://github.com/mristin/ski-leu-desktop",
    author="Marko Ristin",
    author_email="marko@ristin.ch",
    classifiers=[
        # yapf: disable
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9'
        # yapf: enable
    ],
    license="License :: OSI Approved :: MIT License",
    keywords="ski game body pose estimation",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "black==22.12.0",
            "mypy==0.991",
            "pylint==2.15.8",
            "coverage>=6.5.0,<7",
            "twine",
            # We always have trouble installing Pyinstaller 5.7.0 on Windows.
            "pyinstaller<5.7.0",
            "pillow==9.3.0",
        ],
    },
    py_modules=["skileu"],
    packages=find_packages(
        exclude=["tests", "continuous_integration", "dev_scripts", "experimental"]
    ),
    package_data={
        "skileu": [
            "media/images/*",
            "media/sfx/*",
        ]
    },
    data_files=[
        (".", ["LICENSE", "README.rst", "requirements.txt"]),
    ],
    entry_points={
        "console_scripts": [
            "ski-leu=skileu.main:entry_point",
        ]
    },
)
