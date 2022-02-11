import os
import yaml
import argparse
import csv
import torch
import torch.nn as nn
import tqdm
from dataset.data_reader import DataReader
from dataset.dataloader import loader
from tabulate import tabulate

from models.MlModels import BIOregressor
from models.NNmodel import BIONNregressor
from torch.utils.tensorboard import SummaryWriter


def main_NN(cfg):

    # Load data
    train_loader, valid_loader, test_loader = loader(cfg)

    # Define device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Define loss
    f_loss = nn.MSELoss()

    # Define model
    model = BIONNregressor(cfg["PREPROCESSING"]["NUM_COMPONNENT"])

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["MODEL"]["FCN"]["LR"])

    # Tracking with tensorboard
    tensorboard_writer = SummaryWriter(log_dir=cfg["TENSORBOARD"])

    # Training loop
    for epoch in range(50):
        print(f"EPOCH : {epoch}")

        model.train()

        n_samples = 0
        tot_loss = 0.0

        for inputs, targets in tqdm.tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute the forward pass through the network up to the loss
            outputs = model(inputs)
            loss = f_loss(outputs, targets)

            n_samples += inputs.shape[0]
            tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Training Loss : {tot_loss}")

        with torch.no_grad():

            model.eval()
            n_samples = 0
            tot_loss = 0.0

            for inputs, targets in tqdm.tqdm(valid_loader):

                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                n_samples += inputs.shape[0]
                tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()

        print(f"Validation Loss : {tot_loss}")

    # Testing loop
    for inputs in tqdm.tqdm(test_loader):
        output = model(inputs)


def main_ML(cfg):
    """Main pipeline to train an SVM, RandomForest, Linear Model

    Args:
        cfg (dict): Config parameters
    """

    # Load Data
    data = DataReader(cfg)
    data.read_process_data()
    X_train, X_valid, Y_train, Y_valid = data.get_train_valid()
    X_test, ID = data.get_test()

    # Define the model
    BioRegressor = BIOregressor(cfg)
    print(BioRegressor.get_params())
    # Train the model
    BioRegressor.train_grid_search(X_train, X_valid, Y_train, Y_valid)

    # Get the performances
    R2_train, R2_valid = BioRegressor.get_r2()

    # Run inference on the test set
    test_prediction = BioRegressor.inference_grid_search(X_test)

    # Print the regression coeffs
    print(
        tabulate(
            [["R2 Training Set", R2_train], ["R2 Validation Set", R2_valid]],
            tablefmt="orgtbl",
        )
    )
    print(BioRegressor.get_params())

    # Write Prediction to a CSV file
    file = open(cfg["MODEL_OUTPUT"], "w")

    # create the csv writer
    writer = csv.writer(file)

    # write the header of the csv file
    writer.writerow(["_ID", "0"])

    for i, id in enumerate(ID):
        writer.writerow([id, test_prediction[i]])

    # close the file
    file.close()


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

    # If model is NN
    if config_file["MODEL"]["FCN"]["ACTIVE"]:
        main_NN(config_file)

    # If model is ML model
    else:
        main_ML(config_file)
