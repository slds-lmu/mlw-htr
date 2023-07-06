"""MLW-OCR Projekt.

Lectiomat Class for the Medieval Latin Dictionary.
The class includes an end-to-end pipeline for the
lemma-detection and the OCR task.
"""

from functools import reduce
from typing import Callable, List, Union

import numpy as np
import torch
from PIL import Image
from transformers import (AutoImageProcessor, PreTrainedTokenizerFast,
                          VisionEncoderDecoderModel)
from ultralytics.yolo.engine.results import Results
from ultralyticsplus import YOLO


class Lectiomat:
    """Main Class for mlw-lectiomat library."""

    def __init__(
        self,
        model: str = "misoda/htr-mlw-best",
        path_img_seg: str = "misoda/yolo-mlw",
    ):
        """Instantiate Class.

        :param model: Model string for the OCR model (hugggingface hub).
        :param path_img_seg: Model string for the YOLO model (huggingface hub)
        """
        self.model: YOLO = YOLO(path_img_seg)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model)
        self.image_processor = AutoImageProcessor.from_pretrained(
            "microsoft/swin-tiny-patch4-window7-224"
        )
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained(model)
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def _filter_upper_max_bb(self, bbs: List[List[List[float]]]) -> list:
        """Filter Data.

        Find all instances with more than one bounding box and return
        as a DataFrame.

        :param bbs: List of bounding boxes in batch.
        :returns: Largest bounding box in the upper left.
        """
        x_bound: float = 0.5
        y_bound: float = 0.25
        get_largest_a: Callable = lambda elem: elem[
            np.argmax(list(map(lambda e: (e[2] - e[0]) * (e[3] - e[1]), elem)))
        ]
        outliers_invert: Callable = lambda elem: np.invert(
            list(map(lambda e: e[0] > x_bound or e[1] > y_bound, elem))
        )
        filtered: list = list(map(lambda e: e[outliers_invert(e)], bbs))
        return list(map(get_largest_a, filtered))

    def _cut_out(self, path: str, bounding_boxes: torch.tensor) -> np.ndarray:
        """Cut Bounding Boxes Out.

        :param path: Path to image.
        :param bounding_boxes: One bounding box obtained by the model as a tuple.s

        :returns: Cut out images as a numpy array.
        """
        bb: list = list(map(lambda e: float(e.item()), bounding_boxes))
        img: Image = np.asarray(Image.open(path))
        y, x, _ = np.shape(img)
        bb[0] = int(bb[0] * x)
        bb[1] = int(bb[1] * y)
        bb[2] = int(bb[2] * x)
        bb[3] = int(bb[3] * y)
        return img[bb[1] : bb[3], bb[0] : bb[2], :]

    def _get_bbs(self, imgs: List[str]) -> List[torch.Tensor]:
        """Apply Model and Get Bounding-Boxes.

        :param imgs: List of paths to images in a batch.
        :returns: List of bounding-boxes for each element in the batch.
        :raises Exception: Exception if lemma is not found on image.
        """
        results: List[Results] = self.model(imgs)
        bbs: list = list(map(lambda e: e.boxes.xyxyn, results))

        if reduce(
            lambda acc, e: e == 0 or acc, list(map(lambda e: len(e), bbs)), False
        ):
            imgs_w_bb: list = list(
                zip(*list(filter(lambda e: len(e[0]) == 0, zip(bbs, imgs))))
            )
            ind: list = list(imgs_w_bb[1])
            raise Exception(f"ERROR -> No Lemma found on images: {ind}!")
        if reduce(lambda acc, e: e > 1 or acc, list(map(lambda e: len(e), bbs)), False):
            bbs = self._filter_upper_max_bb(bbs)
        if bbs[0].size()[0] == 1:
            bbs[0] = bbs[0].squeeze()

        return bbs

    def apply_img_seg(self, imgs: List[str]) -> np.ndarray:
        """Apply Image Detection Model.

        Apply image detection model on image loaded based on provided `imgs` list.
        E.g.:

        ```lm.apply_img_seg(['./953.jpg', './954.jpg'])```

        :param imgs: Paths to images as a list.
        :returns: List of cut out images as numpy arrays.
        """
        assert isinstance(imgs, list), (
            "ERROR -> `img` must be of type list but is of type ",
            str(type(imgs)),
        )
        results: List[torch.Tensor] = self._get_bbs(imgs)
        return list(map(lambda e: self._cut_out(*e), zip(imgs, results)))

    def apply_ocr(self, imgs: np.ndarray) -> List[str]:
        """Apply OCR-Model.

        :param imgs: Cropped image from image segmentation.
        :returns: List of strings with predicted words.
        """
        assert isinstance(imgs, list), (
            "ERROR -> `img` must be of type list but is of type ",
            str(type(imgs)),
        )
        pixel_values = self.image_processor(imgs, return_tensors="pt").pixel_values.to(
            self.device
        )

        generated_ids = self.ocr_model.generate(pixel_values)
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def __call__(self, imgs: Union[str, List[str]]) -> Union[List[str], str]:
        """Run Pipeline.

        :param imgs: Paths to images as a list.
        :returns: List of strings with predicted words.
        :raises Exception: Exception if input type is not of string or a list of strings.
        """
        is_list: bool = True if isinstance(imgs, list) else False
        if not is_list:
            is_atomic: bool = True if isinstance(imgs, str) else False
            if is_atomic:
                assert isinstance(imgs, str)
                return self.apply_ocr(self.apply_img_seg(imgs=[imgs]))[0]
            else:
                raise Exception(
                    (
                        "ERROR -> `img` must be of type list but is of type ",
                        str(type(imgs)),
                    )
                )
        else:
            assert isinstance(imgs, list)
            return self.apply_ocr(self.apply_img_seg(imgs=imgs))
