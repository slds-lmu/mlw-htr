"""MLW-OCR Projekt.

Class to convert the data annotated by OFA to the YOLOv8 PyTorch TXT format.
"""

import os
import sys
import typing

import cv2
import numpy as np
import pandas as pd
import yaml
from data_utils import create_dataset, create_target_folder
from PIL import Image
from tqdm import tqdm


class Output2YOLOv8:
    """Class to Build YOLO-compatible Dataset.

    YOLOv8 requires data in the format of YOLOv5 PyTorch.txt.
    The format is documented at the following resource:
    https://roboflow.com/formats/yolov5-pytorch-txt
    https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
    (Section about manual dataset setup)
    """

    BB_CLASS: int = 0
    YOLO_MAX_WIDTH: int = 640

    def __init__(
        self,
        source2output: str,
        source2data_json: str,
        path2images: str,
        target: str,
        max_width: int = 640,
        seed: int = 42,
        size: int = sys.maxsize,
    ):
        """Initialize Class.

        :param source2output: Path to output file.
        :param source2data_json: Path to data.json file.
        :param path2images: Path to images.
        :param target: Path to target location.
        :param max_width: Max width for downscaling (must be below 640).
        :param seed: Seed for random processes.
        :param size: Desired size of the dataset
        """
        self.source_output: str = source2output
        self.source_data_json: str = source2data_json
        self.source_images: str = path2images
        self.target: str = target
        self.YOLO_MAX_WIDTH: int = (
            max_width if max_width < self.YOLO_MAX_WIDTH else self.YOLO_MAX_WIDTH
        )
        self.seed: int = seed
        self.size: int = size

    def _create_datasets(self) -> None:
        """Create Datasets.

        Wrapper for the `create_dataset` function form `data_utils.py` and slice
        df to desired size.
        """
        self.df: pd.DataFrame = create_dataset(
            self.source_data_json, self.source_output, shuffle=True, seed=self.seed
        )
        self.df = self.df[0 : self.size]

    def _write_yaml(self) -> None:
        """Write dataset.yaml file.

        It is only necessary to provide the path to the image folder since
        YOLO extracts the name of the images to import the label.
        """
        file_out: typing.IO = open(os.path.join(self.target, "dataset.yaml"), "w")
        # TODO: Put into parameters
        output_dict: dict = {
            "path": self.target,  # Path to root folder
            "train": "train/images",  # Train images relative to `path`
            "val": "val/images",  # Val images relative to `path`
            "test": "test/images",  # Test images relative to `path`
            "names": {self.BB_CLASS: "lemma"},  # Classes
        }
        yaml.dump(output_dict, file_out, default_flow_style=False)
        file_out.close()

    def _create_dir_structure(self) -> None:
        """Create Dataset Structure.

        Creates base folder as well as train and test folders within.
        Both folders include directories for images and labels furthermore.
        """
        # Create base dir
        create_target_folder(self.target)
        # Create `train` dir
        create_target_folder(os.path.join(self.target, "train"))
        create_target_folder(os.path.join(self.target, "train", "images"))
        create_target_folder(os.path.join(self.target, "train", "labels"))
        # Create `val` dir
        create_target_folder(os.path.join(self.target, "val"))
        create_target_folder(os.path.join(self.target, "val", "images"))
        create_target_folder(os.path.join(self.target, "val", "labels"))
        # Create `test` dir
        create_target_folder(os.path.join(self.target, "test"))
        create_target_folder(os.path.join(self.target, "test", "images"))
        create_target_folder(os.path.join(self.target, "test", "labels"))

    def _downscale_instance(self, img: np.ndarray) -> np.ndarray:
        """Downscale Image and Information.

        :param img: Image to be downscaled.
        :returns: Downsized image.
        """
        height, width, _ = img.shape

        ratio: float = self.YOLO_MAX_WIDTH / width

        target_width: int = int(width * ratio)
        target_height: int = int(height * ratio)

        target_dim: tuple = (target_width, target_height)
        resized_img: np.ndarray = cv2.resize(
            img, target_dim, interpolation=cv2.INTER_AREA
        )

        return resized_img

    def _create_bb_data(
        self, img: np.ndarray, x: int, y: int, width: int, height: int
    ) -> str:
        """Create Labels for BB.

        Computes relative properties for bounding boxes, that are returned as
        comma-separated values as a string ready to be written to file.

        :param img: Image to get shape from.
        :param x: X-position of BB.
        :param y: Y-position of BB.
        :param width: Width of BB.
        :param height: Height of BB.

        :returns: Comma-separated values as a string.
        """
        height_total, width_total, _ = img.shape
        x_rel: float = x / width_total
        y_rel: float = y / height_total
        width_rel: float = width / width_total
        height_rel: float = height / height_total
        return " ".join(map(str, (self.BB_CLASS, x_rel, y_rel, width_rel, height_rel)))

    def _write_instance(
        self, name: str, path: str, img: np.ndarray, label: str
    ) -> None:
        """Write one Instance.

        Writes an instance to `labels` and `images` folder.

        :param name: Name of the instance (e.g. <id>.jpg / <id>.txt)
        :param path: Path where instance will be saved.
        :param img: Numpy array of the image.
        :param label: Comma-separated values as a string, ready to be written
            to file.
        """
        # Save .txt file
        out_file: typing.IO = open(
            os.path.join(path, "labels", str(name) + ".txt"), "w"
        )
        out_file.write(label)
        out_file.close()

        # Save .jpg file
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(path, "images", str(name) + ".jpg"), img)

    def _write_df(self, df: pd.DataFrame, path: str) -> None:
        """Write Instance to Directory.

        Iterate over DataFrame and convert numbers to relative format.

        :param df: DataFrame of instances to be written to directory (musts be
            in format of the df obtained in `combine_datasets`)
        :param path: Parent folder of `images` and `labels` folders (not root-dir
            but train/test/val)
        """
        for _, row in tqdm(df.iterrows()):
            img: np.ndarray = np.asarray(
                Image.open(
                    os.path.join(self.source_images, str(row["id"]) + ".jpg"), mode="r"
                )
            )

            label: str = self._create_bb_data(
                img, row["x"], row["y"], row["width"], row["height"]
            )
            img = self._downscale_instance(img)

            self._write_instance(row["id"], path, img, label)

    def _prepare(self) -> None:
        """Prepare Class to Build Dataset.

        Preparation before calling `build_dataset()`.
        """
        self._create_datasets()
        self._create_dir_structure()
        self._write_yaml()

    def build_dataset(self, test_train_p: float, val_p: float = 0.05) -> None:
        """Build Dataset.

        :param test_train_p: Parameter to determine approximate train-test-split \
            (e.g. 0.1 for training size with portion of 10%).
        :param val_p: Parameter to determine the size of the validation set \
            which is sliced off the training set. Default set to 0.05.
        """
        self._prepare()

        # Divide dataset to into train and test
        data: pd.DataFrame = self.df.sample(frac=1, random_state=self.seed)
        bound_train_test: int = int(test_train_p * len(data))
        data_test: pd.DataFrame = data[0:bound_train_test]
        data_train: pd.DataFrame = data[bound_train_test::]

        # Slice off training data
        bound_val: int = int(val_p * len(data_train))
        data_val: pd.DataFrame = data_train[0:bound_val]
        data_train = data_train[bound_val::]

        self._write_df(data_test, os.path.join(self.target, "test"))
        self._write_df(data_val, os.path.join(self.target, "val"))
        self._write_df(data_train, os.path.join(self.target, "train"))
