"""MLW-OCR Projekt.

Bayerische Akademie der Wissenschaften.
LMU München

Gilary Vera Nuñez, Philipp Koch
Esteban Garces Arias, Dr. Matthias Aßenmacher, Prof. Dr. Christian Heumann

2023

Dataset class file for main OCR model.
"""

import logging
import os
from typing import Any, Callable, Dict, Union

import pandas as pd
import torch
from augmentations import Augmentation
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, ViTFeatureExtractor


class MLWDataset(Dataset):
    """Dataset Class for MLW.

    :param path_to_data_json: Path to `data.json` in MLW folder.
    :param path_to_images: Path to images (lemmata, not original images).
    :param tokenizer: Tokenizer used to tokenize the dataset.
    :param feature_extractor: Feature extractor used for visual encoder.
    :param max_target_length:
    """

    def __init__(
        self,
        df: pd.DataFrame,
        path_to_images: str,
        tokenizer: Union[PreTrainedTokenizerFast, Any],
        feature_extractor: Union[ViTFeatureExtractor, Any],
        augmentation: bool = False,
        random_erasing: bool = True,
        random_rotation: bool = True,
        color: bool = True,
        max_target_length: int = 32,
        debug: bool = False,
    ):
        """Instantiate Class.

        :param df: DataFrame from `data.json`.
        :param path_to_images: Path to image folder.
        :param tokenizer: Tokenizer used to tokenize lemmata.
        :param feature_extractor: Feature extractor for image transformer.
        :param augmentation: Boolean variable wether augmentation should be applied.
            WARNING: If set to true, increase amount of epochs!
        :param random_erasing: Apply random erasing in the augmentation pipeline.
        :param random_rotation: Apply random rotating in the augmentation pipeline.
        :param color: Apply modification to color, sharpness and blur image.
        :param max_target_length: Max length for padding.
        :param debug: Enables debug mode which scales down the dataset.
        """
        # self.path2djson: str = path_to_data_json
        self.path2imgs: str = path_to_images
        self.tokenizer: Union[PreTrainedTokenizerFast, Any] = tokenizer
        self.feat_extr: Union[ViTFeatureExtractor, Any] = feature_extractor
        self.max_trgt_len: int = max_target_length
        self.debug: bool = debug

        self.df: pd.DataFrame = df
        if self.debug:
            self.df = self.df.head(20)
            logging.info("`MLWDataset` truncated (debug mode).")
        self.check_exists()
        if augmentation:
            self.aug_prob: float = 0.5
        else:
            self.aug_prob = 1.0
        self.augmentation: Augmentation = Augmentation(
            random_erasing=random_erasing, random_rotation=random_rotation, color=color
        )
        self.tensor_trafo: Callable = transforms.ToTensor()
        logging.info("`MLWDataset` initialized.")

    def check_exists(self) -> None:
        """Check if all file exist.

        Main dataframe is truncated by all missing images.
        """
        rm_list: list = []
        for _, row in tqdm(self.df.iterrows(), desc="Check if all images exist."):
            path: str = os.path.join(self.path2imgs, str(row["id"]) + ".jpg")
            if not os.path.exists(path):
                rw: str = row["id"]
                rm_list.append(rw)
                # logging.info(f"{rw}.jpg does not exist")
        self.df = self.df[~self.df["id"].isin(rm_list)]
        n: int = len(self.df)
        logging.info(f"Dataset verified! Size: {n}")

    def __len__(self) -> int:
        """Return Length of Dataset.

        :returns: Length of dataset as int.
        """
        return len(self.df)

    def __getitem__(self, index: Union[int, list]) -> Dict[str, torch.tensor]:
        """Get Item.

        Get item at position `index`.
        :param index: Index to return element from.
        :returns: Image and label.
        """
        path2img: str = os.path.join(
            self.path2imgs, str(self.df["id"].iloc[index]) + ".jpg"
        )
        img: Image = Image.open(path2img).convert("RGB")
        if torch.rand(1).item() > self.aug_prob:
            img = self.augmentation(img)
            img[img > 1.0] = 1.0
            img[img < 0.0] = 0.0
        pixel_values: torch.Tensor = self.feat_extr(
            img, return_tensors="pt"
        ).pixel_values

        lemma: str = self.df["lemma"].iloc[index]
        labels: torch.Tensor = self.tokenizer(
            lemma, padding="max_length", max_length=self.max_trgt_len
        ).input_ids

        labels = [
            label if label != self.tokenizer.pad_token_id else -100 for label in labels
        ]
        encoding: dict = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels),
        }
        return encoding
