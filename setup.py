"""Setup script for chatscope package."""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="chatscope",
    version="1.0.0",
    author="22wojciech",
    author_email="wojciech@example.com",
    description="Analyze and categorize ChatGPT conversation exports using OpenAI API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/22wojciech/chatscope",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'chatscope=chatscope.cli:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Utilities",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "plotting": [
            "matplotlib>=3.5.0",
        ],
    },

    keywords="chatgpt openai conversation analysis categorization nlp",
    project_urls={
        "Bug Reports": "https://github.com/22wojciech/chatscope/issues",
        "Source": "https://github.com/22wojciech/chatscope",
        "Documentation": "https://github.com/22wojciech/chatscope#readme",
    },
)