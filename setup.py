"""
Setup configuration for OmniParser package.

This enables installation of the omnitool package as an editable installation:
    pip install -e .

This makes absolute imports work throughout the codebase.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="omniparser",
    version="0.1.0",
    author="Microsoft OmniParser Team",
    description="OmniParser - Web UI Parsing with Multi-Modal AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/OmniParser",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "gradio>=4.0.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "openai>=1.0.0",
        "anthropic>=0.7.0",
        "groq>=0.4.0",
        "pytest>=7.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
