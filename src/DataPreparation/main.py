"""MLW-OCR Projekt.

Main entry point to run dataset factory for YOLOv8 PyTorch TXT format.
"""

import hydra
from omegaconf import DictConfig
from Output2YOLOv8 import Output2YOLOv8


@hydra.main(
    version_base=None, config_path="../../config/DataFactory", config_name="config"
)
def main(cfg: DictConfig) -> None:
    """Start Data Factory.

    :param cfg: Configuration for Hydra. DO NOT PASS ARGUMENT!
    """
    test = Output2YOLOv8(
        source2output=cfg.paths.data_to_output,
        source2data_json=cfg.paths.data_to_data_json,
        path2images=cfg.paths.data_to_imgs,
        target=cfg.paths.target_dir,
        max_width=cfg.properties.max_width,
        seed=cfg.properties.seed,
        size=cfg.properties.size,
    )
    test.build_dataset(0.1)


if __name__ == "__main__":
    main()
