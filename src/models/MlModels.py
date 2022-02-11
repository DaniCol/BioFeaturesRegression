import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


class BIOregressor:
    def __init__(self, cfg):
        """Init the model class

        Args:
            cfg (dict): Parameters of the config
        """

        self.cfg = cfg["MODEL"]
        self.r2_train = 0
        self.r2_valid = 0

        if self.cfg["SVM"]["ACTIVE"]:
            self.model = SVR(
                kernel=self.cfg["SVM"]["KERNEL"],
                degree=self.cfg["SVM"]["DEGREE"],
                gamma=self.cfg["SVM"]["GAMMA"],
                coef0=self.cfg["SVM"]["COEFF0"],
                tol=self.cfg["SVM"]["TOL"],
                C=self.cfg["SVM"]["C"],
                epsilon=self.cfg["SVM"]["EPS"],
                shrinking=True,
            )
            if self.cfg["SVM"]["GRID_SEARCH"]["ACTIVE"]:
                self.params = {
                    "kernel": self.cfg["SVM"]["GRID_SEARCH"]["KERNEL"],
                    "degree": self.cfg["SVM"]["GRID_SEARCH"]["DEGREE"],
                    "gamma": self.cfg["SVM"]["GRID_SEARCH"]["GAMMA"],
                    "coef0": self.cfg["SVM"]["GRID_SEARCH"]["COEFF0"],
                    "tol": self.cfg["SVM"]["GRID_SEARCH"]["TOL"],
                    "C": self.cfg["SVM"]["GRID_SEARCH"]["C"],
                    "epsilon": self.cfg["SVM"]["GRID_SEARCH"]["EPS"],
                }
                self.grid_search = GridSearchCV(
                    self.model, self.params, verbose=2, cv=3
                )

        if self.cfg["RANDOMFOREST"]["ACTIVE"]:
            self.model = RandomForestRegressor(
                n_estimators=self.cfg["RANDOMFOREST"]["N_ESTIMATOR"],
                max_depth=self.cfg["RANDOMFOREST"]["MAX_DEPTH"],
            )
            if self.cfg["RANDOMFOREST"]["GRID_SEARCH"]["ACTIVE"]:
                self.params = {
                    "n_estimators": list(
                        range(*self.cfg["RANDOMFOREST"]["GRID_SEARCH"]["N_ESTIMATOR"])
                    ),
                    "max_depth": list(
                        range(*self.cfg["RANDOMFOREST"]["GRID_SEARCH"]["MAX_DEPTH"])
                    ),
                }
                self.grid_search = GridSearchCV(
                    self.model, self.params, verbose=2, cv=3
                )

        if self.cfg["LINEAR"]["ACTIVE"]:
            self.model = LinearRegression(fit_intercept=self.cfg["LINEAR"]["NORM"])

    def train(self, X_train, X_valid, Y_train, Y_valid):
        """Trains the model

        Args:
            X_train (np.array): training features
            X_valid (np.array): valid features
            Y_train (np.array): training ground truth
            Y_valid (np.array): valid ground truth
        """

        # Train the model
        print("Sart Training")
        self.model.fit(X_train, Y_train)
        self.r2_train = self.model.score(X_train, Y_train)

        # Check the performances on the validation set
        self.r2_valid = self.model.score(X_valid, Y_valid)
        print("End of Training")

    def train_grid_search(self, X_train, X_valid, Y_train, Y_valid):
        """Function to train a model with grid search optim
        """
        X_train = np.concatenate((X_train, X_valid))
        Y_train = np.concatenate((Y_train, Y_valid)).ravel()
        # Train with grid search
        print("Sart Training with grid search")
        self.grid_search.fit(X_train, Y_train)
        self.r2_train = self.grid_search.score(X_train, Y_train)

        # Check the performances on the validation set
        self.r2_valid = self.grid_search.score(X_valid, Y_valid)
        print("End of Training with grid search")

    def inference_grid_search(self, X_test):
        """Run inference with the best found model

        Args:
            X_test (np.array): test features

        Returns:
            np.array: estimated value for Y_test
        """

        # Run inference on the test set
        prediction = self.grid_search.predict(X_test)

        return prediction

    def inference(self, X_test):
        """Runs inference on the test set

        Args:
            X_test (np.array): test features

        Returns:
            np.array: estimated value for Y_test
        """

        # Run inference on the test set
        prediction = self.model.predict(X_test)

        return prediction

    def get_params(self):
        """Returns:
            np.array: parameters of the fit model
        """
        return self.model.get_params()

    def get_r2(self):
        """Returns:
            tuple of int: the train and valid R2 coeef
        """
        return self.r2_train, self.r2_valid

    def get_grid_search_params(self):
        return self.grid_search.get_params()

    def get_model(self):
        """This function will be usefull to plot the regression trees

        Returns:
            sklearn.model: the fitted model
        """
        return self.model
