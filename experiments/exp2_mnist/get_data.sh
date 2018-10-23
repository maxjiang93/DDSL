#!/bin/bash
mkdir data
mkdir data/MNIST
mkdir data/polyMNIST
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz 
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz 
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz 
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz 
mv train-images-idx3-ubyte.gz data/MNIST/train-images-idx3-ubyte.gz
mv train-labels-idx1-ubyte.gz data/MNIST/train-labels-idx1-ubyte.gz
mv t10k-images-idx3-ubyte.gz data/MNIST/t10k-images-idx3-ubyte.gz
mv t10k-labels-idx1-ubyte.gz data/MNIST/t10k-labels-idx1-ubyte.gz