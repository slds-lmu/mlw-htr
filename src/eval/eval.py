"""MLW-OCR Projekt.

Script for evaluating the OCR-model.
"""

import argparse
import logging
import os
import re

import easyocr
import pandas as pd
import pytesseract
import torch
from paddleocr import PaddleOCR
from PIL import Image
from tqdm.auto import tqdm
from transformers import (AutoImageProcessor, PreTrainedTokenizerFast,
                          VisionEncoderDecoderModel)


class Evaluate:
    """Parent Class for Evaluation."""

    def __call__(self, image: Image):
        """Call Method.

        :param image: Image on which OCR will be applied.
        :raises NotImplementedError: if called.
        """
        raise NotImplementedError()


class EvaluateOurs(Evaluate):
    """Class for Evaluation of Our Model."""

    def __init__(self, tokenizer_path: str, model_path: str):
        """Instantiate Class.

        :param tokenizer_path: Path to tokenizer.json file.
        :param model_path: Path to model folder.
        """
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        self.feature_extractor = AutoImageProcessor.from_pretrained(
            "microsoft/swin-tiny-patch4-window7-224"
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, path: str) -> str:
        """Use Our Model for Images.

        :param path: Path to the image.
        :returns: String containg the detected word.
        """
        image = Image.open(path).convert("RGB")
        pixel_values = self.feature_extractor(
            image, return_tensors="pt"
        ).pixel_values.to(self.device)
        # autoregressively generate caption (uses greedy decoding by default)
        generated_ids = self.model.generate(pixel_values)
        generated_text: str = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return generated_text


class EvaluateTesseract(Evaluate):
    """Class for Evaluation of Our Tesseract."""

    def __call__(self, path: str) -> str:
        """Use Tesseract for images.

        :param path: Path to the image.
        :returns: String containg the detected word.
        """
        output: str = pytesseract.image_to_string(path, lang="lat")
        output = re.sub(r"\n|\s|\f", "", output)
        return output


class EvaluateEasyOCR(Evaluate):
    """Class for Evaluation of Our EasyOCR."""

    def __init__(self) -> None:
        """Instantiate Class."""
        self.reader = easyocr.Reader(["la"])

    def __call__(self, path: str) -> str:
        """Use EasyOCR for images.

        :param path: Path to the image.
        :returns: String containg the detected word.
        """
        result: str = self.reader.readtext(path)
        return result[0][-2]


class EvaluatePaddleOCR(Evaluate):
    """Class for Evaluation of Our PaddleOCR."""

    def __init__(self) -> None:
        """Instantiate Class."""
        self.reader = PaddleOCR(lang="la")
        logging.basicConfig(level="warning")

    def __call__(self, path: str) -> str:
        """Use PaddleOCR for images.

        :param path: Path to the image.
        :returns: String containg the detected word.
        """
        result: str = self.reader.ocr(path, cls=True)
        return str(result[0][0][-1][0])


def init_model(mode: str, tokenizer_path: str, model_path: str) -> Evaluate:
    """Initialize Evaluator Model.

    :param mode: Mode which model should be evaluated.
    :param tokenizer_path: Path to tokenizer.json file.
    :param model_path: Path to model folder.
    :returns: Evalutor object for the specific evaluation.
    :raises NotImplementedError: if mode is not set correctly.
    """
    if mode == "ours":
        return EvaluateOurs(tokenizer_path, model_path)
    if mode == "tesseract":
        return EvaluateTesseract()
    if mode == "easyocr":
        return EvaluateEasyOCR()
    if mode == "paddleocr":
        return EvaluatePaddleOCR()
    else:
        raise NotImplementedError(f"{mode}-Evaluator is not Implemented!")


def main(
    mode: str = "ours",
    path_data_json: str = "data_test_test.json",
    path_data: str = "./data/interim/lemmata_img/images",
    tokenizer_path: str = "./models/tokenizer/MLW_Tokenizer.json",
    model_path: str = "./models/AUG-SWIN-GPT2",
    target_path: str = "test_predictions.json",
) -> None:
    """Run Evaluation.

    :param mode: Mode which model should be evaluated.
    :param path_data_json: Path to data.json.
    :param path_data: Path to image directory.
    :param tokenizer_path: Path to tokenizer.json file.
    :param model_path: Path to model folder.
    :param target_path: Path to save predictions as json file.
    """
    df = pd.read_json(path_data_json)

    progress_bar = tqdm(range(len(df)))
    collector: dict = {"id": [], "lemma": [], "prediction": []}
    evaluator: Evaluate = init_model(mode, tokenizer_path, model_path)
    for row in df.iterrows():
        id = row[1][0]
        lemma = row[1][1]
        path = os.path.join(path_data, str(id) + ".jpg")
        try:
            # image = Image.open(path).convert("RGB")
            generated_text: str = evaluator(path)
            collector["id"].append(id)
            collector["lemma"].append(lemma)
            collector["prediction"].append(generated_text)
        except Exception:
            logging.info(f"Loading image failed: {id}.jpg")
        progress_bar.update(1)

    results = pd.DataFrame(collector)
    results.to_json(target_path)


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="Evaluation Script.",
        description="Evaluate different models on the test dataset.",
    )
    parser.add_argument(
        "--mode",
        required=False,
        default="ours",
        help="Mode which model should be evaluated {'ours', 'tesseract', 'easyocr'}",
    )
    parser.add_argument(
        "--data-json",
        required=False,
        default="data_test.json",
        help="Path to `data.json` file.",
    )
    parser.add_argument(
        "--trgt-path",
        required=False,
        default="test_predictions.json",
        help="Path where target file will be written to",
    )
    args: argparse.Namespace = parser.parse_args()
    main(mode=args.mode, path_data_json=args.data_json, target_path=args.trgt_path)
