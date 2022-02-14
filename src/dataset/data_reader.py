import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import os
import argparse

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tabulate import tabulate


class DataReader:
    """A class to read the csv file and preprocess the data.
    """

    def __init__(self, cfg):
        """Init class method

        Args:
            cfg (dict): config parameters
        """
        # Init config
        self.cfg = cfg

        # Init train, valid and test set
        self.X_train = None
        self.Y_train = None
        self.X_valid = None
        self.Y_valid = None
        self.X_test = None
        self.test_features = None
        self.features = None
        self.reduced_features = None

        # Init PCA
        if self.cfg["PREPROCESSING"]["PCA"]:
            self.pca = PCA(n_components=self.cfg["PREPROCESSING"]["NUM_COMPONNENT"])

        # Define normaliser
        if self.cfg["PREPROCESSING"]["NORMALIZATION"]["MINMAX"]:
            self.scaler = MinMaxScaler()
        if self.cfg["PREPROCESSING"]["NORMALIZATION"]["MEANSTD"]:
            self.scaler = StandardScaler()

    def read_process_data(self):
        """A function to read the CSV data

        Args:
            cfg (dict): config file
        """

        # Load the training and test data with Pandas
        self.features = pd.read_csv(
            os.path.join(self.cfg["DATA_DIR"], "input_training_F4pMRVn.csv")
        )
        labels = pd.read_csv(
            os.path.join(self.cfg["DATA_DIR"], "output_training_4fUZmFS.csv")
        )
        self.test_features = pd.read_csv(
            os.path.join(self.cfg["DATA_DIR"], "input_testing.csv")
        )

        # Remove first column _ID
        self.features.drop("_ID", axis=1, inplace=True)
        labels.drop("_ID", axis=1, inplace=True)
        self.X_test = self.test_features.drop("_ID", axis=1)

        # Normalize
        if self.cfg["PREPROCESSING"]["NORMALIZATION"]["ACTIVE"]:
            self.features = self.scaler.fit_transform(self.features)
            self.X_test = self.scaler.transform(self.X_test)

        # Run PCA on the input features and the test set
        if self.cfg["PREPROCESSING"]["PCA"]:
            self.reduced_features = self.pca.fit_transform(self.features)
            self.X_test = self.pca.transform(self.X_test)

        else:
            self.reduced_features = self.features

        # Split the data
        self.X_train, self.X_valid, self.Y_train, self.Y_valid = train_test_split(
            self.reduced_features,
            labels,
            test_size=self.cfg["PREPROCESSING"]["VALID_RATIO"],
            random_state=self.cfg["PREPROCESSING"]["RANDOM_STATE"],
        )

    def get_train_valid(self):
        """Get training and validation set

        Returns:
            numpy arrays: the data
        """
        return (
            self.X_train,
            self.X_valid,
            self.Y_train.to_numpy(),
            self.Y_valid.to_numpy(),
        )

    def get_test(self):
        """Getter to return the test data

        Returns:
            Numpy array: the test data
        """
        return self.X_test, self.test_features.iloc[:, 0]

    def get_features(self):
        """Get the features

        Returns:
            Numpy array: features
        """
        return self.features

    def __str__(self):
        """Print method overriden

        Returns:
            str: a description of the data with the size of each set
        """
        return tabulate(
            [
                ["Features", self.features.shape],
                ["Selected Features", self.reduced_features.shape],
                ["Training", self.X_train.shape],
                ["Valid", self.X_valid.shape],
                ["Test", self.X_test.shape],
            ],
            headers=["Set", "Size"],
            tablefmt="orgtbl",
        )

    def plot_pca_variance(self,num_components=295,cumulative=False):
        """A function to visualize pca variance

        Args:
            num_components (int): number of components 
            cumulative (bool): whether we plot the cumulative variance or by components
        """
        if cumulative :
            plt.plot(np.arange(1,num_components+1),np.cumsum(self.pca.explained_variance_ratio_[0:num_components]))
            plt.title("Eplained Cumulative variance % by componant",size=18)
            plt.xlabel("Number of Components",size = 14)
            plt.ylabel("% Variance Explained", size=14)
        else :
            plt.hist(self.pca.explained_variance_ratio_[0:num_components])
            plt.title("Eplained variance % by componant",size=18)
            plt.xlabel("Component #",size = 14)
            plt.ylabel("% Variance Explained", size=14)
        
        plt.show()
    
    def plot_missing_ratio(self):
        """plots missing ratio
        """
        na_df = ((self.features==0).sum() / len(self.features)) * 100 
        na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio %' :na_df})
        missing_data.plot(kind = "barh")
        plt.show()

if __name__ == "__main__":
    # Init the parser;
    inference_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Add path to the config file to the command line arguments;
    inference_parser.add_argument(
        "--path_to_config",
        type=str,
        required=True,
        default="./config.yaml",
        help="path to config file.",
    )
    args = inference_parser.parse_args()

    # Load config file
    with open(args.path_to_config, "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

    # Read data
    data = DataReader(config_file)
    data.read_process_data()
    print(data)
