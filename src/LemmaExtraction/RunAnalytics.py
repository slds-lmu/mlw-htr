"""MLW-OCR Projekt.

Main entry point to run image segmentation.
"""

import hydra
from DataSegmentation4Analytics import DataSegmentation4Analytics
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
    seg: DataSegmentation4Analytics = DataSegmentation4Analytics(
        model=model, source_path=cfg.paths.source_dir, target_path=cfg.paths.alytcs_dir
    )
    seg.run_segmentation()


if __name__ == "__main__":
    main()
