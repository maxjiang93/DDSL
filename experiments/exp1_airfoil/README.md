## Shape Optimization of 2D Airfoils Using DDSL

### Processing data
(In progress)
``` bash
chmod +x *.sh
```

To process the airfoil data, run the data processing shell script:
```bash
process_data.sh
```

To see sample airfoils from the dataset, run the dataset test python script:
```bash
python3 airfoil_dataset_test.py
```

### Training
To train the model from scratch with default settings, simply run the training shell script which will invoke training with the default input parameters:
```bash
./train.sh
```

### Shape Optimization
To run an example shape optimization, run the optimization shell script:
```bash
./optim.sh
```
