# Retinanet
## Data preprossing
* h52csv.ipynb & h52csv.py : transform the format of dataset from .mat to .csv
## Training
* `python keras-retinanet/keras_retinanet/bin/train.py --batch-size 5 --steps 6700 --epochs 20 --random-transform csv train/train_data.csv train/class.csv`
  * use `--batch-size 5` to set up the batch size
  * use `--steps 6700` to set up the tje number of steps per epoch
  * use `--epochs 20` to set up the epochs
  * use `csv train/train_data.csv train/class.csv` to set up the training dataset
## Testing
* keras-retinanet/examples/ResNet50RetinaNet.ipynb or ResNet50RetinaNet.py : test the testing image and restore the result in the format of .json
