"""MLW-OCR Projekt.

File for utilty functions.
"""

import functools
import typing

import pandas as pd
import torch
from datasets import Dataset
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)
from torchmetrics import CharErrorRate
from transformers import PreTrainedTokenizerFast


class CERMetric:
    """CER-Class.

    Class to Compute Character-Error-Rate and keep necessary
    properties for efficient computing.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        """Instantiate Class.

        Compute Character-Error-Rate.

        :param tokenizer: Tokenizer, to decode predictions and labels.
        """
        self.tokenizer: PreTrainedTokenizerFast = tokenizer
        self.metric: CharErrorRate = CharErrorRate()

    def __call__(self, pred: torch.Tensor) -> typing.Dict[str, float]:
        """Compute Character Error Rate.

        :param pred: Predictions
        :return: Dict of {'cer': <computed cer value>}
        """
        labels_ids: torch.Tensor = pred.label_ids
        pred_ids: torch.Tensor = pred.predictions

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        cer: float = self.metric(pred_str, label_str)

        return {"cer": cer}


# TODO: Remove
def train_tokenizer(
    lemma_list_path: str, tokenizer_path: str, batch_size: int = 1000
) -> Tokenizer:
    """Train Tokenizer.

    Load corpus, split words to characters and train tokenizer on them.
    Tokenizer will also be saved at a specified ('tokenizer path')
    location.

    :param lemma_list_path: Path to lemma .xlsx file.
    :param tokenizer_path: Path where tokenizer is to be saved.
    :param batch_size: Batch size for training.
    :returns: Trained tokenizer
    """
    corpus: pd.DataFrame = pd.read_excel(lemma_list_path)["Lemmata"].to_frame()
    rm_list: list = []
    for e in corpus["Lemmata"].values:
        if isinstance(e, float):
            rm_list.append(e)
    corpus = corpus[~corpus["Lemmata"].isin(rm_list)]
    corpus["list"] = list(map(lambda e: list(e), corpus["Lemmata"].values))
    corpus["text"] = list(map(lambda e: " ".join(e), corpus["list"].values))

    dataset = Dataset.from_pandas(corpus)

    alphabet: set = set(functools.reduce(lambda x, y: x + y, corpus["list"], []))

    tokenizer: Tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    special_tokens_map = {
        "cls_token": "<CLS>",
        "pad_token": "<PAD>",
        "sep_token": "<SEP>",
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<UNK>",
    }

    tokenizer.add_special_tokens(list(special_tokens_map))

    def batch_iterator() -> typing.Generator[pd.DataFrame, None, None]:
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["text"]

    # Train tokenizer
    trainer = trainers.BpeTrainer(
        vocab_size=len(alphabet), special_tokens=list(special_tokens_map)
    )  # Check recommendations for vocabulary size

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # Post-processor and decoder
    tokenizer.post_processor = processors.ByteLevel(
        trim_offsets=False,
    )
    tokenizer.decoder = decoders.ByteLevel()

    tokenizer.save(tokenizer_path)
    return tokenizer
