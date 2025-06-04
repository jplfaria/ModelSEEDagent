#!/usr/bin/env python3

from setuptools import find_packages, setup

# Read requirements from requirements.txt
with open("requirements.txt", "r") as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="modelseed-agent",
    version="0.1.0",
    description="ModelSEED Agent: AI-powered metabolic modeling with LangGraph workflows",
    author="ModelSEED Team",
    author_email="support@modelseed.org",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "modelseed-agent=src.cli.standalone:app",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    include_package_data=True,
    zip_safe=False,
)
