"""MLW-OCR Projekt.

Class using a trained YOLO model to write all results to a dedicated file.
"""

import json
import logging
import os
import pathlib
import shutil
import typing
from typing import Any

import pandas as pd
import ultralytics
from tqdm import tqdm


class DataSegmentation4Analytics:
    """Class for Data-Segmentation and return Information about BBs..

    Used to find all BBs using YOLOv8.
    """

    MISSING: str = "MISSING"

    def __init__(self, model: ultralytics.YOLO, source_path: str, target_path: str):
        """Instantiate Class.

        :param model: Image-detection model (only `ultralytics.YOLO` supported).
        :param source_path: Path to source folder ('zettel').
        :param target_path: Path to target folder (set in config.yaml).
        """
        self.model: ultralytics.YOLO = model

        self.report_list: list = []

        f: typing.IO = open(os.path.join(source_path, "data.json"))
        self.files = pd.DataFrame(json.load(f))["id"]
        f.close()
        self.source_path: str = source_path

        self.target_path: str = target_path
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        else:
            shutil.rmtree(target_path)
            os.makedirs(self.target_path)

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

        Run Model and return report.
        """
        f: typing.IO = open(os.path.join(self.target_path, "yolo_analytics.txt"), "a")

        for _, elem in tqdm(enumerate(self.files)):
            img_path: str = os.path.join(self.source_path, "zettel", str(elem) + ".jpg")
            results: Any = None
            try:
                results = self.model(img_path)[0]
            except Exception:
                logging.warn(f"Image not found: {img_path}")
            if results is None:
                self.report_list.append(self.MISSING)
                f.write(str(elem) + ", " + self.MISSING + "\n")
                continue

            bb_list: list = []
            for result in results.boxes.xyxyn:
                bb: tuple = tuple(map(lambda e: float(e.item()), result))
                bb_list.append(bb)
            conf_list: list = list(map(lambda e: float(e.item()), results.boxes.conf))
            f.write(str(elem) + ", " + str((conf_list, bb_list)) + "\n")
        f.close()

    def run_segmentation(self) -> None:
        """Run All Commands."""
        self.apply_model()
        self.return_report()
