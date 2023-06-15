"""Setup file for package installation."""

from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="scanna",
    version="0.0.2",
    author="A. Ali Heydari, Oscar Davalos",
    author_email="aliheydari@ucdavis.edu",
    description=
    "scANNA: single-cell ANalysis using Neural Attention",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/SindiLab/scANNA",
    download_url="https://github.com/SindiLab/scANNA",
    packages=find_packages(),
    install_requires=[
        "tqdm>=4.47.0",
        "adabelief-pytorch>=0.2.0",
        "torch>=1.13",
        "scanpy>=1.7.0",
        "tensorboardX>=2.1",
        "prettytable",
    ],
    classifiers=[
        "Development Status :: 1 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT Software License",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
        ":: Bioinformatics :: Deep Learning"
    ],
    keywords=
    "Single Cell RNA-seq, Automatic Classification, Attention-Neural Networks,"
    "Deep Learning, Transfer Learning")
