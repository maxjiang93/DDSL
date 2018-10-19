# polymnist
Code for running PolyFFT tests on MNIST

## Get MNIST data
To get the original MNIST data, run the bash script `get_data.sh`. This will automatically create the directory `data/` and all requried subdirectories.
```bash
bash get_data.sh
```

## Process MNIST data into polygons
```bash
python3 process_mnist.py
```

## Train CNN (Variant of LeNet that allows flexible input size)
`train_mnist.py` takes in various command line inputs. Check source code for useage.
```bash
python3 train_mnist.py -s 32 -e phys -l runs/exp0 -j 12 -o result.csv
```
The example above trains it with input of 32 by 32, uses phys encoding, stores tensorboard output to `runs/exp0`, uses 12 threads for dataloading, and writes final test result to result.csv.
