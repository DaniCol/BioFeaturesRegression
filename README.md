# PROJET : Algorithmes en Data Sciences
Daniel Colombo, Hamza Benslimane, Mathieu Nalpon 

# Pipeline 

## Data Preparation

There are 2 ways to load the data. The first one is with the following script : data_reader.py.
If you want to use classical machine learning methods you can simply use this script to load the data.
However, if you want yo use torch models, the data are load n a torch manner in the dataloader.py script.

## Dimension Reduction

To reduce the dimension of our features, we can use a simple PCA the parameters of which can be found in the config file.

## Normalise 

There is the possibility to normalise the data with the MinMax method or the standart deviation. However, the data were already normalised between -1 and 1 so only the standart deviation is needed (in order to run an SVM). 

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

After each training, we plot the predicted value with respect to the ground truth value. This way you can see how close the prediction of the regression is to the ground truth. 

### Grid Search

The grid search can be used to find the best parameters for a SVM and for a Random Forest. To train with grid search, check the config file and set RANDOM FOREST - GRID SEARCH - ACTIVE to true (if you want to train the grid search) make sure to set all the other model to False. 

# RUN the training

Go to ./src and run the following command.
Don't forget to change the MODEL_OUTPUT name depending on the model you want to train.

```
cd src
python3 main.py --path_to_config ./config.yaml
```
# RUN the average

One can run a model averaging by specifying the names of the CSV files in the config file that you want to average. 
To do so, modify in the config the PATHS variable
```
cd src
python3 average.py --path_to_config ./config.yaml
```


# LIEN DE LA VIDEO 
https://youtu.be/wkJ1TQOyfxs
