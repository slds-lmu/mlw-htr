"""MLW-OCR Projekt.

Main entry point to run image segmentation.
"""

import hydra
from DataSegmentation import DataSegmentation
from omegaconf import DictConfig
from ultralytics import YOLO


@hydra.main(
    version_base=None, config_path="../../config/DataSegmentation", config_name="config"
)
def main(cfg: DictConfig) -> None:
    """Start Data Segmentation.

    :param cfg: Configuration for Hydra. DO NOT PASS ARGUMENT!
    """
    model: YOLO = YOLO(cfg.paths.yolo_model)
    seg: DataSegmentation = DataSegmentation(
        model=model, source_path=cfg.paths.source_dir, target_path=cfg.paths.target_dir
    )
    seg.run_segmentation()


if __name__ == "__main__":
    main()
