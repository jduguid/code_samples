from setuptools import setup, find_packages


setup(
    name="outlier_detection",
    version="0.1",
    description="Toy examples of outlier detection routines for use as code samples.",
    packages=find_packages(),
    url="https://github.com/jduguid/code_samples",
    author="James Duguid",
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
    ],
)
