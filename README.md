# PROJET : Algorithmes en Data Sciences

## Data Preparation

There are 2 ways to load the data. The first one is with the following script : data_reader.py.
If you want to use classical machine learning methods you can simply use this script to load the data.
However, if you want yo use torch models, the data are load n a torch manner in the dataloader.py script.

## Dimension Reduction

To reduce the dimension of our features, we can use a simple PCA the parameters of which can be found in the config file.
We can test other methods like LDA or filtering methods ..  . . . . . . .

## Models

### SVM

To tune the SVM, check the config file and set 'ACTIVE' to True

### Random Forest

To tune the Random Forest, check the config file and set 'ACTIVE' to True

### Linear Regression

To tune the Linear Regression, check the config file and set 'ACTIVE' to True

### FCNetwork

To tune the FCNetwork, check the config file and set 'ACTIVE' to True

### Data and Results Visualization

à Implémenter

### Grid Search for svm/pca

à implémenter

### XGBoost

à tester


# RUN the training

Go to ./src and run the following command.
Don't forget to change the MODEL_OUTPUT name depending on the model you want to train.

```
cd src
python3 main.py --path_to_config ./config.yaml
```
