from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression


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

        if self.cfg["RANDOMFOREST"]["ACTIVE"]:
            self.model = RandomForestRegressor(
                n_estimators=self.cfg["RANDOMFOREST"]["N_ESTIMATOR"],
                max_depth=self.cfg["RANDOMFOREST"]["MAX_DEPTH"],
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

    def get_model(self):
        """This function will be usefull to plot the regression trees

        Returns:
            sklearn.model: the fitted model
        """
        return self.model
