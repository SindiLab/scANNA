# scANNA (Package Repository)


This repository hosts the package for [scANNA: single-cell ANalysis using Neural Attention](https://icml-compbio.github.io/2022/papers/WCBICML2022_paper_18.pdf) (Preprint, submitted). To make package development and maintaining more efficient, we have located training scripts and tutorials in different repositories into different repositories, as listed below.

![scANNA_Diagram](scANNA_Diagram.png)

[![arXiv:10.48550/arXiv.2206.04047](http://img.shields.io/badge/arXiv-110.48550/arXiv.2206.04047-A42C25.svg)](https://doi.org/10.48550/arXiv.2206.04047)

## Installing scANNA
### Installing the GitHub Repository (Recommended)
scANNA can be installed using PyPI:
```
$ pip install git+https://github.com/SindiLab/scANNA.git
```
or can be first cloned and then installed as the following:
```
$ git clone https://github.com/SindiLab/N-ACT.git
$ pip install ./scANNA
```

### Install Package Locally with `pip`
Once the files are available, make sure to be in the same directory as `setup.py`. Then, using `pip`, run:

````bash
pip install -e .
````
In the case that you want to install the requirements explicitly, you can do so by:
````bash
pip install -r requirements.txt
````
Although the core requirements are listed directly in `setup.py`. Nonetheless, it is good to run this beforehand in case of any dependecies conflicts.

## Training scANNA
All main scripts for training (and finetuning) our deep learning model are located in the `training_and_finetuning_scripts` folder in this repository.

## [Tutorials](https://github.com/SindiLab/Tutorials/tree/main/scANNA)
We have compiled a set of notebooks as tutorials to showcase scANNA's capabilities and interptretability. These notebooks located [here](https://github.com/SindiLab/Tutorials/tree/main/scANNA). 

**Please feel free to open issues for any questions or requests for additional tutorials!**

## Trained Models
TODO: Will be released with the next preprint for scANNA.
## Citation
If you found our work useful for your ressearch, please cite our preprint:

```
Coming Soon!
```
