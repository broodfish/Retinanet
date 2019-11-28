# Retinanet
## Data preprossing
* h52csv.ipynb & h52csv.py : transform the format of dataset from .mat to .csv
## Training
* `python keras-retinanet/keras_retinanet/bin/train.py --image-min-side 100 --image-max-side 166 --batch-size 5 --steps 6700 --epochs 20 --random-transform csv train/train_data.csv train/class.csv`
## Testing
* keras-retinanet/examples/ResNet50RetinaNet.ipynb or ResNet50RetinaNet.py : test the testing image and restore the result in the format of .json
