# ANN Regressor
This repo contains Artificial Neural Network model for regression analysis.
Also the program compares results with [MLP Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) 
model provided by Scikit-learn package.

## Installation
Install required packages:
```shell script
pip install -r requirements.txt
```

## Usage
Generate a dataset before using the model.
```shell script
python3 dataset_generator.py -f [DATASET_FILE] -l [DATASET_LENGTH]
```

Run the model with generated dataset:
```shell script
python3 nn_regression.py -f [DATASET_FILE]
```

You can also tune hidden layer size by passing arguments from the command line:
```shell script
python3 nn_regression.py -f [DATASET_FILE] -H 100
```