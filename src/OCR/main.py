"""MLW-OCR Projekt.

Main file for training and evaluation of the OCR-model.
"""

import logging
import math
import os
import pathlib
import shutil
from typing import Union

import datasets
import hydra
import numpy as np
import pandas as pd
import torch
from MLWDataset import MLWDataset
from omegaconf import DictConfig
from PIL import Image
from transformers import (AutoImageProcessor, BeitConfig, BeitImageProcessor,
                          BeitModel, GPT2Config, GPT2LMHeadModel,
                          PreTrainedTokenizerFast, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, SwinConfig, SwinModel,
                          Trainer, TrainingArguments,
                          VisionEncoderDecoderConfig,
                          VisionEncoderDecoderModel, ViTConfig,
                          ViTImageProcessor, ViTModel, default_data_collator)
from transformers.image_processing_utils import BaseImageProcessor
from utils import CERMetric

VIT: str = "ViT"
SWIN: str = "SWIN"
BEIT: str = "BEIT"

PROCESSOR_DICT: dict = {
    VIT: ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k"),
    SWIN: AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224"),
    BEIT: BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k"),
}

CONFIG_DICT: dict = {VIT: ViTConfig(), SWIN: SwinConfig(), BEIT: BeitConfig()}

ENCODER_DICT: dict = {
    VIT: ViTModel(ViTConfig()),
    SWIN: SwinModel(SwinConfig()),
    BEIT: BeitModel(BeitConfig()),
}


def pre_train_decoder(
    tokenizer_path: str,
    data_path: str,
    pre_train_model_path: str,
    epochs: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    block_size: int = 32,
) -> None:
    """Train Decoder.

    :param tokenizer_path: Path to tokenizer json file.
    :param data_path: Path to training corpora.
    :param pre_train_model_path: Name under which the model will be saved.
    :param epochs: Number of epochs the model is trained on.
    :param per_device_train_batch_size: Batch size for training.
    :param per_device_eval_batch_size: Batch size for evaluation.
    :param block_size: Block size for truncation and padding.
    """
    if os.path.exists(
        pre_train_model_path
    ):  # check if pre-trained model already exist at target
        logging.info("Pre-Trained model exists. Continue.")
        return
    else:
        config_encoder = CONFIG_DICT["SWIN"]
        config_decoder = GPT2Config()
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
            config_encoder, config_decoder
        )
        model = VisionEncoderDecoderModel(config=config)
        model.decoder.save_pretrained("TMP")

        model = GPT2LMHeadModel.from_pretrained("TMP")
        shutil.rmtree("TMP")

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
            # TODO into config
            pre_train_model_path,
            num_train_epochs=epochs,
            save_strategy="no",
            evaluation_strategy="epoch",
            per_device_train_batch_size=per_device_train_batch_size,  # 192,
            per_device_eval_batch_size=per_device_eval_batch_size,  # 192,
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


@hydra.main(version_base=None, config_path="../../config/OCR", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train OCR-Model.

    :param cfg: Configuration for Hydra. DO NOT PASS ARGUMENT!
    """
    # Technical Setup
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["MLFLOW_FLATTEN_PARAMS"] = "true"
    torch.manual_seed(cfg.global_config.seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(cfg.global_config.seed)

    # Log
    logging.info(f"Goal: {cfg.global_config.goal}")

    target_path: str = os.path.join(
        cfg.model_configs.target_folder, cfg.model_configs.model_name
    )

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg.paths.tokenizer)
    feature_extractor: BaseImageProcessor = PROCESSOR_DICT[cfg.model_configs.vis_model]

    data_json_path: Union[str, list] = cfg.paths.lemma_datajson
    if isinstance(data_json_path, str):
        df: pd.DataFrame = pd.read_json(cfg.paths.lemma_datajson)
        df = df.sample(frac=1).reset_index(drop=True)
        df_train: pd.DataFrame = df.iloc[0 : int(cfg.data.tr_te * len(df))]
        df_test: pd.DataFrame = df.iloc[int(cfg.data.tr_te * len(df)) : len(df)]
    else:
        logging.info("Load seperate train and test file.")
        df_train = pd.read_json(data_json_path[0])
        logging.info(f"Training file of length {len(df_train)} loaded.")
        df_test = pd.read_json(data_json_path[1])
        logging.info(f"Training file of length {len(df_test)} loaded.")
        if len(df_test) > len(df_train):
            Warning("Test dataset is larger than training dataset!")

    dataset_train: MLWDataset = MLWDataset(
        df=df_train,
        path_to_images=cfg.paths.path_images,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        augmentation=cfg.training_configs.augmentation,
        random_erasing=cfg.training_configs.rand_eras,
        random_rotation=cfg.training_configs.rand_rota,
        color=cfg.training_configs.color,
        max_target_length=cfg.model_configs.max_len,
        debug=cfg.global_config.debug,
    )
    dataset_test: MLWDataset = MLWDataset(
        df=df_test,
        path_to_images=cfg.paths.path_images,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        max_target_length=cfg.model_configs.max_len,
        debug=cfg.global_config.debug,
    )

    # Load architectures in the model
    config_encoder = CONFIG_DICT[cfg.model_configs.vis_model]
    config_decoder = GPT2Config()

    # Group architectures and define model
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
        config_encoder, config_decoder
    )
    if cfg.global_config.goal == "train":
        if cfg.pre_training.apply:
            pre_train_decoder(
                tokenizer_path=cfg.paths.tokenizer,
                data_path=cfg.pre_training.pre_train_corpus,
                pre_train_model_path=cfg.pre_training.decoder_path,
                epochs=cfg.pre_training.epochs,
                per_device_train_batch_size=cfg.pre_training.per_device_train_batch_size,
                per_device_eval_batch_size=cfg.pre_training.per_device_eval_batch_size,
            )
            encoder = ENCODER_DICT[cfg.model_configs.vis_model]
            encoder.save_pretrained("TMP")
            model: VisionEncoderDecoderModel = (
                VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                    "TMP", cfg.pre_training.decoder_path
                )
            )
            logging.info("Pre-Trained Vision-Encoder Model loaded successful!")
            shutil.rmtree("TMP")
        else:
            model = VisionEncoderDecoderModel(config=config)
    elif cfg.global_config.goal == "eval":
        model = VisionEncoderDecoderModel.from_pretrained(target_path)

    # load a fine-tuned image captioning model and corresponding tokenizer and image processor
    tokenizer.model_max_length = cfg.model_configs.max_len

    # Set up tokenizer and models for task
    special_tokens_dict = {
        "pad_token": "[PAD]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "unk_token": "[UNK]",
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    model.decoder.resize_token_embeddings(len(tokenizer))
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Set Beam-Search Params
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.max_length = cfg.nlg.max_length
    model.config.early_stopping = cfg.nlg.early_stopping
    model.config.no_repeat_ngram_size = cfg.nlg.no_repeat_ngram_size
    model.config.length_penalty = cfg.nlg.length_penalty
    model.config.num_beams = cfg.nlg.num_beams

    use_fp16: bool = torch.cuda.is_available()
    training_args: Seq2SeqTrainingArguments = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        num_train_epochs=cfg.training_configs.epochs,
        per_device_train_batch_size=cfg.training_configs.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training_configs.per_device_eval_batch_size,
        fp16=use_fp16,
        output_dir="./",
        logging_steps=cfg.training_configs.logging_steps,
        save_steps=cfg.training_configs.save_steps,
        eval_steps=cfg.training_configs.eval_steps,
        report_to=cfg.training_configs.report_to,
        run_name=cfg.training_configs.run_name,
    )

    cer_fun: CERMetric = CERMetric(tokenizer)

    trainer: Seq2SeqTrainer = Seq2SeqTrainer(
        model=model,
        tokenizer=feature_extractor,
        args=training_args,
        compute_metrics=cer_fun,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        data_collator=default_data_collator,
    )
    if cfg.global_config.goal == "train":
        trainer.train()
        metrics: dict = trainer.evaluate()
        logging.info(metrics)

        # Save Model
        pathlib.Path(cfg.model_configs.target_folder).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(target_path)
        logging.info(f"Saved model at: {cfg.model_configs.target_folder}")

        # TODO:Remove
        # let's perform inference on an image
        image = Image.open(os.path.join(cfg.paths.path_images, "959.jpg")).convert(
            "RGB"
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pixel_values = feature_extractor(image, return_tensors="pt").pixel_values.to(
            device
        )

        # autoregressively generate caption (uses greedy decoding by default)
        generated_ids = model.generate(pixel_values)
        generated_text = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        logging.info("Generated Text: <" + str(generated_text) + ">")
    elif cfg.global_config.goal == "eval":
        metrics = trainer.evaluate()
        logging.info(metrics)


if __name__ == "__main__":
    main()
