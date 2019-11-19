## DDSL: Deep Differential Simplex Layer for Neural Networks

<img src="./doc/teaser.png" alt="DDSL_teaser" width=400>

### Introduction
In this project, we present a novel neural network layer that performs differentiable rasterization of arbitrary simplex-mesh-based geometrical signals (e.g., point clouds, line mesh, triangular mesh, tetrahedron mesh, polygon and polyhedron) of arbitrary dimensions. We further provide examples of incorporating the DDSL into neural networks for tasks such as polygonal image segmentation and neural shape optimization (for MNIST digits and airfoils).

Our deep learning code base is written using [PyTorch 1.0](https://pytorch.org/) in Python 3, in conjunction with standard Python packages such as [Numpy](http://www.numpy.org/). PyTorch version > 1.0 is required.

### Using the DDSL layer for your applications
We provide an efficient natively PyTorch-based implementation of the DDSL. Detailed documentation for APIs can be found in [ddsl/ddsl.py](ddsl/ddsl.py). For examples on using the DDSL implementation for rasterizing a given input mesh, refer to the jupyter notebooks in the folder [examples](./examples).

### Experiments
To replicate the experiments in our paper, please refer to codes in the [experiments](./experiments) folder. Detailed instructions for each experiment can be found in the corresponding directories.

### Related Projects
This code base contains code for the two projects below. The DDSL layer is a differentiable version for the one outlined in the ealier paper.
- Jiang, Chiyu, Dana Lynn Ona Lansigan, Philip Marcus, and Matthias Nießner. "[DDSL: Deep Differentiable Simplex Layer for Learning Geometric Signals.](https://arxiv.org/abs/1901.11082)" The IEEE International Conference on Computer Vision (ICCV), 2019.
- Jiang, Chiyu, Dequan Wang, Jingwei Huang, Philip Marcus, and Matthias Nießner. "[Convolutional Neural Networks on non-uniform geometrical signals using Euclidean spectral transformation.](https://arxiv.org/abs/1901.02070)" International Conference on Learning Representations (ICLR), 2019.

### Cite
Please cite our work if you find it helpful.
```
@InProceedings{Jiang_2019_ICCV,
author = {Jiang, Chiyu "Max" and Lansigan, Dana and Marcus, Philip and Niessner, Matthias},
title = {DDSL: Deep Differentiable Simplex Layer for Learning Geometric Signals},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}

@inproceedings{jiang2018convolutional,
title={Convolutional Neural Networks on Non-uniform Geometrical Signals Using Euclidean Spectral Transformation},
author={Chiyu Max Jiang and Dequan Wang and Jingwei Huang and Philip Marcus and Matthias Niessner},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=B1G5ViAqFm},
}
```
