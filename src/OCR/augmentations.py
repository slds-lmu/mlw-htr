"""MLW-OCR Projekt.

Script including functions regarding augmentations.
"""
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torchvision import transforms


def increase_brightness(img: list, value: int = 20) -> np.ndarray:
    """Increase Brightness of Image.

    :param img: Image to be transformed.
    :param value: TODO
    :returns: Transformed image.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim: int = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def contra_sharp(img: list) -> np.ndarray:
    """Increase Contrast and Sharpen Image.

    :param img: Image to be transformed.
    :returns: Transformed image.
    """
    # Contrast factor
    contr_factor = 5  # 1 = do nothing. We increase contrast

    # Sharpness factor
    sharp_factor = 5  # increase sharpness

    # data_contrast = Image.open(f"./PreProc1/{img}")

    enhancer_contrast = ImageEnhance.Contrast(img)
    image_contrast = enhancer_contrast.enhance(contr_factor)

    enhancer_sharpness = ImageEnhance.Sharpness(image_contrast)
    image_sharp = enhancer_sharpness.enhance(sharp_factor)

    final = image_sharp

    return final


class Augmentation:
    """Class for Different Augmentation Techniques.

    Different augmentation techniques are applied randomly to distort
    the images for training. This class can be used on-the-fly while
    training.
    """

    def __init__(
        self,
        random_erasing: bool = True,
        random_rotation: bool = True,
        color: bool = True,
    ):
        """Instantiate Class.

        :param random_erasing: Apply random erasing on the
            augmentation pipelines.
        :param random_rotation: Apply random rotating on
            the augmentation pipeline.
        :param color: Apply modification to color, sharpness
            and blur image.
        """
        p_rand_err: float = 0.5 if random_erasing else 0
        p_rand_rot: int = 10 if random_rotation else 0
        self.augment: dict = dict()
        # All
        if color:
            augment_0: Callable = torch.nn.Sequential(
                transforms.RandomAffine(degrees=p_rand_rot),
                transforms.GaussianBlur(21),
                transforms.ColorJitter(0.5, 0.5, 0.5),
                transforms.RandomAdjustSharpness(0.5),
                transforms.RandomErasing(p=p_rand_err),
            )
            self.augment[0] = torch.jit.script(augment_0)

            # Blurring
            augment_1: Callable = torch.nn.Sequential(
                transforms.RandomAffine(degrees=p_rand_rot),
                transforms.GaussianBlur(21),
                transforms.RandomAdjustSharpness(0.5),
                transforms.RandomErasing(p=p_rand_err),
            )
            self.augment[1] = torch.jit.script(augment_1)

            # Color Adjustments
            augment_2: Callable = torch.nn.Sequential(
                transforms.RandomAffine(degrees=p_rand_rot),
                transforms.ColorJitter(0.5, 0.5, 0.5),
                transforms.RandomAdjustSharpness(0.5),
                transforms.RandomErasing(p=p_rand_err),
            )
            self.augment[2] = torch.jit.script(augment_2)
        else:
            augment: Callable = torch.nn.Sequential(
                transforms.RandomAffine(degrees=p_rand_rot),
                transforms.RandomErasing(p=p_rand_err),
            )
            self.augment[0] = torch.jit.script(augment)
        self.converter: Callable = transforms.ToTensor()

    def __call__(self, img: Image) -> torch.Tensor:
        """Apply Augmentations.

        :param img: PIL Image to be transformed.
        :returns: Transformed image.
        """
        choice: int = np.random.choice(range(len(self.augment.keys())))
        return self.augment[choice](self.converter(img))
