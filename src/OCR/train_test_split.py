"""MLW-OCR Projekt.

Write train and test data.json to file.
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml


def main() -> None:
    """Run Main Function."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="Train-Test-File Creator",
        description="Program that creates to different data.jsons for testing and training.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to .yaml configuration file (necessary for seed and paths).",
    )
    parser.add_argument(
        "--target-folder",
        required=True,
        help="Path to target folder where new data.json files will be stored.",
    )
    args: argparse.Namespace = parser.parse_args()
    config: dict = yaml.safe_load(open(args.config, "r"))
    seed: int = config["global_config"]["seed"]
    path_to_data_json: str = config["paths"]["lemma_datajson"]
    tr_te: float = config["data"]["tr_te"]
    np.random.seed(seed)

    df: pd.DataFrame = pd.read_json(path_to_data_json)
    df = df.sample(frac=1).reset_index(drop=True)
    df_train: pd.DataFrame = df.iloc[0 : int(tr_te * len(df))]
    df_test: pd.DataFrame = df.iloc[int(tr_te * len(df)) : len(df)]
    df_train.to_json(
        os.path.join(args.target_folder, "data_train.json"), orient="records"
    )
    df_test.to_json(
        os.path.join(args.target_folder, "data_test.json"), orient="records"
    )


if __name__ == "__main__":
    main()
