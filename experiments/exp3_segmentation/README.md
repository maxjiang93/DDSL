## Segmentation using PolygonNet

### Downloading and preprocessing data
Two data sources are needed for preprocessing training and testing data: the original cityscapes data, as well as annotated data from PolygonRNN++. Most of the above process have been automated, but permissions to downloading cityscapes and PolygonRNN++ source code needs to be acquired. Please follow the steps below.
* Register for an account on the [cityscapes page](https://www.cityscapes-dataset.com/register/) and remember the username and password.
* Use our provided script for automating data downloading:
``` bash
chmod +x *.sh
./download_cityscapes_raw.sh --username <your.username> --password <your.password>
```
* Sign-up to download PolygonRNN++ scripts [here](http://www.cs.toronto.edu/polyrnn/code_signup/).
* Download and unzip the PolygonRNN++ scripts in the current directory.
* Run the preprocessing script `preprocess_data.py` to preprocess data. Use the `-h` flag to see additional options:
```bash
python preprocess_data.py
```

### Training
To train the model froms scratch with default settings, simply run the training shell script which will invoke training with the default input parameters:
```bash
./train.sh
```

### Test
To test the final scores with the pretrained checkpoint, run the `download_checkpoint.sh` script:
```bash
./download_checkpoint.sh
```
Then run the test script:
```
python test.py
```

### Create visualizations
To create visualizations with predicted polygon masks over the entire input image, run the visualization script:
```
python visualize_results.py
```