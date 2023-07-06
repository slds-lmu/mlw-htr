"""MLW-OCR Projekt.

File including many useful functions for the project.
"""

import json
import os
import typing

import numpy as np
import pandas as pd

MAX_WIDTH: float = 1750 / 2
MAX_HEIGHT: float = 1200 / 2


def check_path_exists(path: str, throw_error: bool) -> bool:
    """Check if Path Exists.

    Check if an object exists at a given path (`path`).

    :param path: Path to file/directory.
    :param throw_error: Determines if an error is thrown if a file/directory already \
        exists.

    :raises Exception: Exception if `throw_error` is set to `True` and path is not \
        found or not in correct format.

    :returns: Boolean that indicates if path exists or not.
    """
    if "~" in path:
        raise Exception(
            f"ERROR:\n\t'-> Absolute path must be provided. Current path is '{path}'."
        )
    exists: bool = os.path.exists(path)
    if exists:
        return True
    else:
        if throw_error:
            raise Exception(f"ERROR:\n\t'-> Path: '{path}' does not exist.")
        else:
            return False


def create_target_folder(path: str, throw_error: bool = False) -> None:
    """Create Folder.

    Creates a folder after checking if a directory at the same places already
    exists.

    :param path: Path to desired location.
    :param throw_error: Determines if an error is thrown if a directory already \
        exists.
    """
    if not check_path_exists(path, throw_error=throw_error):
        os.mkdir(path)


def load_output(path: str) -> pd.DataFrame:
    """Load and Process Output File.

    Loads and processes output file labeled by the previous visual grounding
    model. Besides coordinates of BBs, length and height are also added to
    the DataFrame. All BBs that are larger than one quarter of the approximate
    image size, will be dropped and considered as failed annotations.

    :param path: Path to output file.
    :return: DataFrame of imported output file.
    """
    f: typing.IO = open(str(path), "r")
    output: list = [json.loads(d) for d in f.readlines()]
    f.close()

    id: list = [d["file"].split(".")[0] for d in output]
    x1: list = [d["result"][0]["box"][0] for d in output]
    y1: list = [d["result"][0]["box"][1] for d in output]
    x2: list = [d["result"][0]["box"][2] for d in output]
    y2: list = [d["result"][0]["box"][3] for d in output]

    outputs_bb: pd.DataFrame = pd.DataFrame(
        np.array([id, x1, y1, x2, y2]).T, columns=["id", "x1", "y1", "x2", "y2"]
    )

    outputs_bb["id"] = outputs_bb["id"].astype("int64")
    outputs_bb["x1"] = round(outputs_bb["x1"].astype("float"))
    outputs_bb["y1"] = round(outputs_bb["y1"].astype("float"))
    outputs_bb["x2"] = round(outputs_bb["x2"].astype("float"))
    outputs_bb["y2"] = round(outputs_bb["y2"].astype("float"))

    # Getting the length and height of the Bounding Boxes
    outputs_bb["width"] = outputs_bb["x2"] - outputs_bb["x1"]
    outputs_bb["height"] = outputs_bb["y2"] - outputs_bb["y1"]

    # Getting centre points
    outputs_bb["x"] = round(outputs_bb["width"].astype("float") / 2 + outputs_bb["x1"])
    outputs_bb["y"] = round(outputs_bb["height"].astype("float") / 2 + outputs_bb["y1"])

    # Getting area
    outputs_bb["area"] = outputs_bb["width"] * outputs_bb["height"]

    # Removing all failed annotations
    max_area: int = int(MAX_WIDTH * MAX_HEIGHT)
    outputs_bb = outputs_bb[outputs_bb["area"] < max_area]
    return outputs_bb


def load_data_json(path: str) -> pd.DataFrame:
    """Load data.json.

    Load `data.json`, which contains image ids and labels, file from
    main data ('MLW') directory.

    :param path: Path to data.json in the  '<drive>/MLW' directory.
    :return: DataFrame of imported data.
    """
    f = open(path)
    data = json.load(f)
    data = pd.DataFrame(data)
    return data


def create_dataset(
    path2datajson: str, path2output: str, shuffle: bool = True, seed: int = 42
) -> pd.DataFrame:
    """Create Dataset to Label.

    Function to merge output and data.json datasets. Both datasets are
    first loaded and subsequently merged.

    :param path2datajson: Path to `data.json` file.
    :param path2output: Path to `output` file.
    :param shuffle: Return ordered (False) or shuffled (True) DataFrame.
    :param seed: Seed for shuffle operation.
    :return: Merged Dataset.
    """
    datajson: pd.DataFrame = load_data_json(path2datajson)
    dataoutput: pd.DataFrame = load_output(path2output)
    data: pd.DataFrame = datajson.merge(dataoutput, on="id")
    data = data.sample(frac=1, random_state=seed)
    return data
