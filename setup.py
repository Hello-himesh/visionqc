from setuptools import setup, find_packages

setup(
    name="visionqc",
    version="1.0.0",
    description="Automated Industrial Defect Detection & Quality Inspection Pipeline",
    author="VisionQC Project",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python-headless>=4.8.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
    ],
    entry_points={
        "console_scripts": [
            "visionqc=visionqc.cli:main",
        ],
    },
)
