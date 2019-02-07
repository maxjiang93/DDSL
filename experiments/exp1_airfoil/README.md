## Shape Optimization of 2D Airfoils Using DDSL

### Processing data
Airfoil data from [airfoiltools.com](http://airfoiltools.com/) was used for this experiment. The data includes Selig format .dat files containing the airfoil shapes and lift and drag data (computed using [XFOIL](https://web.mit.edu/drela/Public/web/xfoil/)) at different angles of attack and Reynolds numbers. The data was retrieved using a webscraping script:
``` bash
python3 scraper.py
```

Some post-processing was done to remove data incorrectly recorded by `airfoiltools`. As there was no pattern in the incorrect data, the post-processing was done manually. The cleaned dataset can be downloaded using the data download shell script:
``` bash
chmod +x *.sh
./get_data.sh
```

To process the airfoil data, run the data processing shell script:
```bash
./process_data.sh
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

A model pretrained using the above script can be downloaded using the checkpoint download shell script:
```bash
./get_checkpoint.sh
```


### Shape Optimization
To run an example shape optimization, run the optimization shell script:
```bash
./optim.sh
```
