import csv
import os
import yaml
import argparse
import numpy as np


def average_models(cfg):
    """Function to average different models specified in the config

    Args:
        cfg (dict): config parameters
    """

    # Load Paths
    paths = cfg["AVERAGE"]

    # Init averaged output
    output = np.zeros((359 - 88 + 1, 2))

    # Loop through single outputs
    for path in paths:
        data = np.genfromtxt(path, delimiter=",", skip_header=1)
        output[:, 1] += data[:, 1]

    # Format the output
    output[:, 0] = data[:, 0]
    output[:, 1] = output[:, 1] / len(paths)

    # Write Prediction to a CSV file
    file = open(cfg["MODEL_OUTPUT"], "w")

    # create the csv writer
    writer = csv.writer(file)

    # write the header of the csv file
    writer.writerow(["_ID", "0"])

    for line in output:
        writer.writerow([int(line[0]), line[1]])

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

    average_models(config_file)
