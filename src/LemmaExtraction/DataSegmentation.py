"""MLW-OCR Projekt.

Bayerische Akademie der Wissenschaften.
LMU München

Gilary Vera Nuñez, Philipp Koch
Esteban Garces Arias, Dr. Matthias Aßenmacher, Prof. Dr. Christian Heumann

2023

Class using a trained YOLO model to cut lemmata out.
"""

import json
import logging
import os
import pathlib
import shutil
import typing
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import ultralytics
from tqdm import tqdm


class DataSegmentation:
    """Class for Data-Segmentation.

    Used to cut out lemmata from index cards.
    """

    MISSING: str = "MISSING"
    ZERO_BB: str = "0"
    ONE_BB: str = "1"
    MANY_BB: str = ">1"

    def __init__(self, model: ultralytics.YOLO, source_path: str, target_path: str):
        """Instantiate Class.

        :param model: Image-detection model (only `ultralytics.YOLO` supported).
        :param source_path: Path to source folder ('zettel').
        :param target_path: Path to target folder.
        """
        self.model: ultralytics.YOLO = model

        self.report_list: list = []

        f: typing.IO = open(os.path.join(source_path, "data.json"))
        self.files = pd.DataFrame(json.load(f))["id"]
        f.close()
        self.source_path: str = source_path

        self.target_path: str = os.path.join(target_path, "images")
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        else:
            shutil.rmtree(target_path)
            os.makedirs(self.target_path)

    def _cut_out(self, path: str, bounding_boxes: torch.tensor) -> np.ndarray:
        """Cut Bounding Boxes Out.

        :param path: Path to image.
        :param bounding_boxes: One bounding box obtained by the model as a tuple.s

        :returns: Cutted images as a numpy array.
        """
        bb: tuple = tuple(map(lambda e: int(e.item()), bounding_boxes))
        img: np.ndarray = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img[bb[1] : bb[3], bb[0] : bb[2], :]

    def return_report(self) -> None:
        """Compose Report and Write to File."""
        assert len(self.files) == len(
            self.report_list
        ), "ERROR:\n\t'-> Did you run `apply_model()` alrady?"
        df: pd.DataFrame = pd.DataFrame({"id": self.files, "success": self.report_list})
        parent: str = str(pathlib.Path(self.target_path).parent.absolute())
        df.to_csv(os.path.join(parent, "report.csv"))

    def apply_model(self) -> None:
        """Apply Model to Each File.

        Logs issues with predictions to logger and to report.
        """
        for _, elem in tqdm(enumerate(self.files)):
            img_path: str = os.path.join(self.source_path, "zettel", str(elem) + ".jpg")
            results: Any = None
            try:
                results = self.model(img_path)[0]
            except Exception:
                logging.warn(f"Image not found: {img_path}")
            if results is None:
                self.report_list.append(self.MISSING)
                continue

            len_bb: int = len(results.boxes.xyxy)
            if len_bb == 0:
                self.report_list.append(self.ZERO_BB)
                logging.info(f"No bounding box found: {len_bb} - File: {img_path}")
                continue
            if len_bb > 1:
                self.report_list.append(self.MANY_BB)
                logging.info(
                    f"More than 1 bounding box found: {len_bb} - File: {img_path}"
                )
                continue
            else:
                self.report_list.append(self.ONE_BB)

            img_cut: np.ndarray = self._cut_out(img_path, results.boxes.xyxy[0])
            img_cut = cv2.cvtColor(img_cut, cv2.COLOR_BGR2RGB)
            file_name: str = os.path.basename(img_path)
            cv2.imwrite(os.path.join(self.target_path, file_name), img_cut)

    def run_segmentation(self) -> None:
        """Run All Commands."""
        self.apply_model()
        self.return_report()
