"""MLW-OCR Projekt.

Script for Generating Training Data for Paired CycleGAN.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm


def generate_image(path: str, lemma: str) -> np.ndarray:
    """Generate Machine-Domain Image.

    :param path: Path to image in original domain.
    :param lemma: String containing the lemma.
    :returns: Triple including the newly generated domain, the original image, and set sum of differences
        in different dimensions.
    """
    # sample text and font
    font = ImageFont.truetype(
        "/home/USER/.local/share/fonts/Affectionately Yours - TTF.ttf",
        256,
        encoding="unic",
    )

    # get the line size
    _, _, text_width, text_height = font.getbbox(lemma)

    # create a blank canvas with extra space between lines
    canvas = Image.new("RGB", (text_width + 40, text_height + 40), "white")

    # draw the text onto the text canvas, and use blue as the text color
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 0), lemma, "black", font)

    # Open original image
    image = Image.open(path)

    # Get ratio of original image of generated image
    ratio = np.array(np.shape(image)[0:2]) / np.array(np.shape(canvas)[0:2])
    x_r, y_r = ratio

    # Get ratio
    mask = np.array(
        [
            np.all(
                np.array((x_r * np.array(np.shape(canvas)[0:2])), dtype=int)
                <= np.array(np.shape(image)[0:2], dtype=int)
            ),
            np.all(
                np.array((y_r * np.array(np.shape(canvas)[0:2])), dtype=int)
                <= np.array(np.shape(image)[0:2], dtype=int)
            ),
        ]
    )
    r = ratio[mask]
    new_canvas = canvas.resize(
        np.flip(np.array((r * np.array(np.shape(canvas)[0:2])), dtype=int)),
        Image.Resampling.LANCZOS,
    )
    h, w = np.array(
        np.divide(
            np.array(np.shape(image)[0:2]) - np.array(np.shape(new_canvas)[0:2]), 2
        ),
        dtype=int,
    )

    # Fill boundaries to match size
    white_canvas = np.uint8(np.ones(np.shape(image), dtype=int) * 255)
    white_canvas[
        h : (np.shape(new_canvas)[0] + h), w : (w + np.shape(new_canvas)[1])
    ] = new_canvas

    # Compute difference between sizes and sum up
    diff: float = np.sum(np.array(np.shape(image)) - np.array(np.shape(new_canvas)))
    return white_canvas, image, diff


def generate_ds(
    df: pd.DataFrame, data_root: str, lemma_img_path: str, target: str, source: str
) -> None:
    """Generate Dataset.

    :param df: DataFrame on which the dataset (train or test) will be based on.
    :param data_root: Path to root of new dataset.
    :param lemma_img_path: Path to lemmata (images).
    :param target: Target domain name.
    :param source: Source dodmain name.
    """
    progress_bar = tqdm(range(len(df)))
    for row in df.iterrows():
        id = row[1][0]
        lemma = row[1][1]
        gen, image, diff = generate_image(
            os.path.join(lemma_img_path, str(id) + ".jpg"), lemma
        )
        # Source is machine
        gen = Image.fromarray(gen)
        source_path: str = os.path.join(data_root, source, str(id) + ".jpg")
        gen.save(source_path)
        # target is lemma
        target_path: str = os.path.join(data_root, target, str(id) + ".jpg")
        image.save(target_path)
        progress_bar.update(1)


def main(
    data_with_deltas_path: str = "data_with_deltas.json",
    max_n: int = 20000,
    data_root: str = "machine2lemma",
    target: str = "lemma",
    source: str = "machine",
    tr_ts: float = 0.9,
    test: str = "test",
    train: str = "train",
    lemmata_img_path: str = "/home/USER/Documents/Uni/WiSe2223/Consulting/mlw-consulting-project/data/interim/lemmata_img/images",
    seed: int = 42,
) -> None:
    """Execute Program.

    :param data_with_deltas_path: Path to data.json including deltas column.
    :param max_n: Total number of rows in final dataset.
    :param data_root: Path to the root of the data folders.
    :param target: Name of the target domain.
    :param source: Name of the source domain.
    :param tr_ts: Train-Test-Split percentage.
    :param test: Name for test folder.
    :param train: Name for train folder.
    :param lemmata_img_path: Path to images.
    :param seed: Seed for random processes.
    """
    np.random.seed(seed)
    df: pd.DataFrame = pd.read_json(data_with_deltas_path)
    df_sorted = df.sort_values(by=["deltas"])
    logging.info(f"Delta at {max_n}: ", str(df.iloc[max_n]))
    df_final: pd.DataFrame = df_sorted.iloc[0:max_n]
    df_final = df_final.sample(frac=1)

    Path(data_root).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(data_root, train)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(data_root, train, source)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(data_root, train, target)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(data_root, test)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(data_root, test, source)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(data_root, test, target)).mkdir(parents=True, exist_ok=True)

    df_train: pd.DataFrame = df_final[0 : int(tr_ts * len(df_final))]
    df_test: pd.DataFrame = df_final[int(tr_ts * len(df_final)) : :]

    # Generate Train split
    generate_ds(
        df_train, os.path.join(data_root, train), lemmata_img_path, target, source
    )

    # Generate Test split
    generate_ds(
        df_test, os.path.join(data_root, test), lemmata_img_path, target, source
    )


if __name__ == "__main__":
    main()
