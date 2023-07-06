"""MLW-OCR Projekt.

Script for Pre-Training of the Decoder Part.
"""

import logging
import math

import datasets
from transformers import (AutoModelForCausalLM, GPT2Config,
                          PreTrainedTokenizerFast, Trainer, TrainingArguments)


def main(
    tokenizer_path: str = "/home/paperspace/mlw-consulting-project/models/tokenizer/MLW_Tokenizer.json",
    data_path: str = "data/processed/latin_words.txt",
    block_size: int = 32,
    pre_train_model_name: str = "pre-trained-decoder",
) -> None:
    """Execute Program.

    :param tokenizer_path: Path to tokenizer json file.
    :param data_path: Path to training corpora.
    :param block_size: Block size for truncation and padding.
    :param pre_train_model_name: Name under which the model will be saved.
    """
    model = AutoModelForCausalLM.from_config(GPT2Config())
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    special_tokens_dict = {
        "pad_token": "[PAD]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "unk_token": "[UNK]",
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    data_raw: datasets.Dataset = datasets.load_dataset(
        "text", data_files=data_path, sample_by="line"
    )

    def pad_texts(e: dict) -> dict:
        """Prepare Input.

        :param e: Element which includes tokenized instances.
        :returns: Prepared dict.
        """
        tokenized = tokenizer(
            e["text"], padding="max_length", max_length=block_size, truncation=True
        )
        return {"labels": tokenized["input_ids"], **tokenized}

    data: datasets.Dataset = data_raw.map(pad_texts, remove_columns=["text"])
    data = data.remove_columns("token_type_ids")

    training_args = TrainingArguments(
        pre_train_model_name,
        num_train_epochs=10,
        save_strategy="no",
        evaluation_strategy="epoch",
        per_device_train_batch_size=192,
        per_device_eval_batch_size=192,
        learning_rate=2e-5,
        weight_decay=0.01,
        report_to="mlflow",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["train"],
    )
    trainer.train()

    eval_results = trainer.evaluate()
    logging.info(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    trainer.save_model()


if __name__ == "__main__":
    main()
