"""MLW-OCR Projekt.

Entry point to clean lemmata corpus and train tokenizer.
"""

import hydra
import pandas as pd
from omegaconf import DictConfig
from TokenizerFunctions import make_corpus, train_tokenizer


@hydra.main(version_base=None, config_path="../../config/OCR", config_name="config")
def main(cfg: DictConfig) -> None:
    """Make Tokenizer.

    :param cfg: Configuration for Hydra. DO NOT PASS ARGUMENT!
    """
    corpus: pd.DataFrame = make_corpus(
        corpus_excel_path=cfg.paths.lemma_excel,
        corpus_datajson_path=cfg.paths.datajson,
        target_path=cfg.paths.lemma_interim_combined,
    )
    train_tokenizer(corpus, cfg.paths.tokenizer)  # TODO: Include batch size in config?


if __name__ == "__main__":
    main()
