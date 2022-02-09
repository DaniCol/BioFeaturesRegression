import argparse
import yaml
import torch
from torch.utils.data import Dataset, DataLoader

from dataset.data_reader import DataReader


class BIO(Dataset):
    """The regression dataset as a pytorch dataset

    Args:
        Dataset (torch.utils.data.Dataset): a torch like dataset
    """

    def __init__(self, X_data, y_data=None, test=False):
        """Init the dataset

        Args:
            X_data (np.array): the features
            y_data (np.array, optional): The labels associated to the labels. Defaults to None for the test set.
            test (bool, optional): A boolean for the test set. Defaults to False.
        """
        self.X_data = X_data
        self.y_data = y_data
        self.test = test

    def __getitem__(self, idx):
        """Get the item at the idx-th position in the dataset

        Args:
            idx (int): index position
        Returns:
            A sample of the dataset
        """

        if self.test:
            return self.X_data[idx]

        return self.X_data[idx], self.y_data[idx]

    def __len__(self):
        """Returns the length of the dataset

        Returns:
            int: length of the dataset
        """
        return self.X_data.shape[0]


def loader(cfg):
    """Function to load the data in pytorch

    Args:
        cfg (dict): parameters of the config

    Returns:
        torch.loaders: the torch loaders for the train, valid and test set
    """

    # Read Data
    data = DataReader(cfg)
    data.read_process_data()
    X_train, X_valid, Y_train, Y_valid = data.get_train_valid()
    X_test = data.get_test()
    print(type(Y_train))
    # Build Torch dataset
    train_dataset = BIO(
        torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float()
    )
    valid_dataset = BIO(
        torch.from_numpy(X_valid).float(), torch.from_numpy(Y_valid).float()
    )
    test_dataset = BIO(torch.from_numpy(X_test).float(), test=True)

    # Build torch loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg["PREPROCESSING"]["BATCH_SIZE"],
        shuffle=True,  # <-- this reshuffles the data at every epoch
        num_workers=cfg["PREPROCESSING"]["NUM_THREADS"],
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=cfg["PREPROCESSING"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["PREPROCESSING"]["NUM_THREADS"],
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=cfg["PREPROCESSING"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["PREPROCESSING"]["NUM_THREADS"],
    )

    return train_loader, valid_loader, test_loader


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

    # Load data
    print(loader(config_file))
