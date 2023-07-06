"""MLW-OCR Projekt.

Script to chose only one final bounding boxes if many bounding boxes were found.
"""
import logging
import os
from typing import IO, Any, Callable

import cv2
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm.auto import tqdm


def get_df(path: str) -> pd.DataFrame:
    """Get DataFrame.

    Read yolo analytics file and construct pandas' DataFrame.
    Additionally filter all instances that have more than one bb.
    :param path: Path.
    :returns: Filtered pandas' DataFrame.
    """
    f: IO = open(path, "r")
    indices: list = []
    confs: list = []
    bbs: list = []
    for line in f.readlines():
        MISSING: Any = None  # noqa: F841,N806
        row = eval(("[" + line + "]"))  # noqa: S307
        index, conf_bb = row[0], row[1]
        conf, bb = conf_bb if conf_bb is not None else ([], [])
        indices.append(str(index))
        confs.append(conf)
        bbs.append(bb)
    f.close()
    df: pd.DataFrame = pd.DataFrame({"id": indices, "confs": confs, "bbs": bbs})
    df["len_bb"] = list(map(lambda e: len(e), df["bbs"].values))
    df = df[df["len_bb"] > 1]
    return df


def cut_out(path: str, bounding_boxes: np.ndarray) -> np.ndarray:
    """Cut Bounding Boxes Out.

    :param path: Path to image.
    :param bounding_boxes: One bounding box obtained by the model as a tuple.s

    :returns: Cutted images as a numpy array.
    """
    img: np.ndarray = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    y, x, _ = img.shape
    x1, y1, x2, y2 = bounding_boxes.tolist()
    x1, y1, x2, y2 = int(x1 * x), int(y1 * y), int(x2 * x), int(y2 * y)
    return img[y1:y2, x1:x2, :]


def save(file_name: str, target_path: str, img_cut: np.ndarray) -> None:
    """Save Image.

    :param file_name: File-name of image (e.g. 954.jpg).
    :param target_path: Target directory, where the image will be saved.
    :param img_cut: Cut out image (by `cut_out` function).
    """
    cv2.imwrite(os.path.join(target_path, file_name), img_cut)


def filter_upper_max_bb(df: pd.DataFrame) -> pd.DataFrame:
    """Filter Data.

    Find all instances with more than one bounding box and return
    as a DataFrame.
    :param df: DataFrame, constructed by `get_df` method.
    :returns: Filtered DataFrame.
    """
    x_bound: float = 0.5
    y_bound: float = 0.25
    get_largest_a: Callable = lambda elem: elem[
        np.argmax(list(map(lambda e: (e[2] - e[0]) * (e[3] - e[1]), elem)))
    ]
    outliers_invert: Callable = lambda elem: np.invert(
        list(map(lambda e: e[0] > x_bound or e[1] > y_bound, elem))
    )
    filtered: list = list(map(lambda e: np.array(e)[outliers_invert(e)], df.bbs.values))
    df_mac_corner: pd.DataFrame = df.copy()
    df_mac_corner["a_largest_corner"] = list(map(get_largest_a, filtered))
    return df_mac_corner


@hydra.main(
    version_base=None, config_path="../../config/DataSegmentation", config_name="config"
)
def main(cfg: DictConfig) -> None:
    """Start Data Segmentation.

    :param cfg: Configuration for Hydra. DO NOT PASS ARGUMENT!
    """
    source_dir: str = cfg.paths.source_dir
    target_path: str = os.path.join(cfg.paths.target_dir, "images")
    alytcs_path: str = os.path.join(cfg.paths.alytcs_dir, "yolo_analytics.txt")
    df: pd.DataFrame = get_df(alytcs_path)
    df = filter_upper_max_bb(df)
    logging.info("Data loaded and filtered.")
    progress_bar: Any = tqdm(range(len(df)))
    for f, bb in zip(df.id.values, df.a_largest_corner.values):
        img_name: str = str(f) + ".jpg"
        img_path: str = os.path.join(source_dir, "zettel", img_name)
        save(img_name, target_path, cut_out(img_path, bb))
        progress_bar.update(1)


if __name__ == "__main__":
    main()
