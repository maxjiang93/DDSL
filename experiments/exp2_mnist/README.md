## Shape Optimization of MNIST Digits Using DDSL

### Downloading and processing data
To download the MNIST data, use the data download script:
``` bash
chmod +x *.sh
./get_data.sh
```

To process the airfoil data, run the data processing shell script:
```bash
python3 process_mnist.py
```

### Training
To train the model from scratch with default settings, simply run the training shell script which will invoke training with the default input parameters:
```bash
./train.sh
```

### Shape Optimization
To run an example shape optimization (a '1' to a '3'), run the optimization shell script:
```bash
./optim.sh
```
