"""VisionQC package setup and distribution configuration."""

from setuptools import setup, find_packages

setup(
    name="visionqc",
    version="1.0.0",
    description="Automated Industrial Defect Detection & Quality Inspection Pipeline",
    long_description="A headless, terminal-executable computer vision quality inspection pipeline for manufacturing components.",
    author="Nagahimesh Vuppala",
    license="MIT",
    packages=find_packages(include=["visionqc", "visionqc.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python-headless>=4.8.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0", "fpdf2>=2.7.0"],
    },
    entry_points={
        "console_scripts": [
            "visionqc=visionqc.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Manufacturing",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
