"""MLW-OCR Projekt.

Functions related to tokenization and tokenizer initialization.
"""

import functools
import typing

import numpy as np
import pandas as pd
from datasets import Dataset
from tokenizers import Tokenizer, decoders, processors
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


def make_corpus(
    corpus_excel_path: str, corpus_datajson_path: str, target_path: str
) -> pd.DataFrame:
    """Prepare Corpus.

    Join excel-lemmata list with lemmata from `data.json`. Function
    cleans list before and returns a `pd.DataFrame` including lemmata
    and respective lemma split into characters for tokenization.

    :param corpus_excel_path: Path to excel file.
    :param corpus_datajson_path: Path to `data.json` file.
    :param target_path: Path where file with all lemmata will be saved.
    :returns: Joined and cleaned DataFrame.
    """
    corpus = pd.read_excel(corpus_excel_path)["Lemmata"].to_frame()
    corpus["Lemmata"] = corpus["Lemmata"].astype(str)

    corpus_data_json = pd.read_json(corpus_datajson_path)
    corpus_data_json = corpus_data_json.rename(columns={"lemma": "Lemmata"})
    corpus_data_json["Lemmata"] = corpus_data_json["Lemmata"].astype(str)
    corpus_data_json = corpus_data_json.drop("id", axis=1)
    corpus_data_json = corpus_data_json.drop_duplicates()

    corpus = pd.concat([corpus, corpus_data_json])

    rm_list: list = []
    for e in corpus["Lemmata"].values:
        if isinstance(e, float):
            rm_list.append(e)

    corpus = corpus[~corpus["Lemmata"].isin(rm_list)]
    corpus["list"] = list(map(lambda e: list(e), corpus["Lemmata"].values))
    np.savetxt(target_path, corpus["Lemmata"].values, fmt="%s")
    return corpus


def train_tokenizer(
    corpus: typing.Union[pd.DataFrame, str], tokenizer_path: str, batch_size: int = 1000
) -> Tokenizer:
    """Train Tokenizer.

    Load corpus and train tokenizer on them.
    Tokenizer will also be saved at a specified ('tokenizer path')
    location.

    :param corpus: Either string to lemmata list file (each lemma on
        a seperate line) or already loaded corpus in form of a
        `pd.DataFrame` with a 'list' column.
    :param tokenizer_path: Path where tokenizer is to be saved.
    :param batch_size: Batch size for training.
    :returns: Trained tokenizer
    """
    if isinstance(corpus, str):
        corpus_tmp: pd.DataFrame = pd.read_csv(corpus)
        corpus_tmp["list"] = list(map(lambda e: list(e), corpus_tmp["Lemmata"].values))
        corpus_tmp["text"] = list(map(lambda e: " ".join(e), corpus_tmp["list"].values))
        corpus = corpus_tmp

    dataset = Dataset.from_pandas(corpus)

    alphabet: set = set(functools.reduce(lambda x, y: x + y, corpus["list"], []))

    special_tokens_list: list = ["[BOS]", "[UNK]", "[EOS]"]

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=len(alphabet) + 10, special_tokens=special_tokens_list
    )
    # tokenizer.pre_tokenizer = Whitespace()

    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()

    tokenizer.post_processor = TemplateProcessing(
        single="$A [EOS]",
        special_tokens=[("[EOS]", 2)],
    )

    dataset = dataset.remove_columns(["list", "__index_level_0__"])
    dataset = dataset.rename_column("Lemmata", "text")

    def batch_iterator() -> typing.Generator[Dataset, None, None]:
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["text"]

    # Train tokenizer
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # Save tokenizer
    tokenizer.save(tokenizer_path)
    return tokenizer
