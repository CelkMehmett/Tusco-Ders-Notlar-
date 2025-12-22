"""
train.py

Basit, tek GPU üzerinde çalışacak HuggingFace Trainer tabanlı eğitim betiği.
Aşağıdaki adımları içerir:
- Konfig okuyup tokenizer/model yükleme
- Veri hazırlama (dataset.prepare_datasets çağrısı)
- Trainer oluşturma ve eğitim
- Model kaydetme ve basit inference demo

Dosyalar:
  - config.py
  - dataset.py

Not: Bu script doğrudan çalıştırılabilir; GPU yoksa hata verebilir.
"""

import os
import time
import logging
import argparse

from datasets import Dataset
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

from config import get_config
from dataset import prepare_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def run_training(cfg):
    # Yükle tokenizer
    logger.info(f"Loading tokenizer: {cfg['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        # T5 gibi modeller pad token içerir; yoksa eos kullan
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # Veri hazırlama
    logger.info("Preparing dataset (this may download from HuggingFace)")
    tokenized = prepare_datasets(cfg, tokenizer)

    # tokenized dict -> HuggingFace Dataset
    dataset = Dataset.from_dict(tokenized)
    logger.info(f"Dataset shape: {len(dataset)} examples")

    # Küçük eval split
    split = dataset.train_test_split(test_size=0.02, seed=cfg.get("seed", 42))
    train_ds = split["train"]
    eval_ds = split["test"]
    logger.info(f"Train/Eval sizes: {len(train_ds)}/{len(eval_ds)}")

    # Model yükle
    logger.info(f"Loading model: {cfg['model_name']}")
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"])
    model.resize_token_embeddings(len(tokenizer))

    n_params = count_parameters(model)
    logger.info(f"Model param count: {n_params:,}")

    # TrainingArguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        fp16=cfg["fp16"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=cfg.get("logging_steps", 50),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        seed=cfg.get("seed", 42),
        predict_with_generate=True,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Eğitim
    t0 = time.time()
    logger.info("Starting training...")
    trainer.train()
    duration = time.time() - t0
    logger.info(f"Training finished in {duration/60:.2f} minutes")

    # Kaydet
    os.makedirs(cfg["output_dir"], exist_ok=True)
    logger.info(f"Saving model and tokenizer to {cfg['output_dir']}")
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    return cfg["output_dir"], duration, n_params, tokenizer


def demo_inference(model_dir, tokenizer, device="cuda"):
    # Basit 5 örnek ve sonuçların basımı
    from transformers import AutoModelForSeq2SeqLM

    logger.info("Loading model for demo inference")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)

    examples = [
        "Summarize the following text: PyTorch is a popular deep learning framework...",
        "Write a short email asking for a meeting to discuss project milestones.",
        "Explain in simple terms what overfitting is and one way to avoid it.",
        "Translate to Turkish: Machine learning models benefit from clean data.",
        "Given a list of items, produce a short bulleted checklist for packing for travel.",
    ]

    for i, prompt in enumerate(examples, 1):
        src = f"### Instruction:\n{prompt}\n\n### Response:\n"
        inputs = tokenizer(src, return_tensors="pt", truncation=True, max_length=512).to(device)
        out = model.generate(**inputs, max_new_tokens=128)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"\n=== Example {i} ===")
        print("Prompt:", prompt)
        print("Output:", text)
        # Basit anlamlılık kontrolü
        if len(text.split()) < 3:
            print("[WARN] Very short output — may be invalid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_demo_only", action="store_true", help="Sadece demo çalıştır")
    args = parser.parse_args()

    cfg = get_config()

    if args.run_demo_only:
        # demo için çıktı klasöründe tokenizer/model olmalı
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg["output_dir"])
        demo_inference(cfg["output_dir"], tokenizer)
        return

    out_dir, duration, n_params, tokenizer = run_training(cfg)
    logger.info(f"Trained model saved to {out_dir}")
    logger.info(f"Training time (s): {duration:.1f}")
    logger.info(f"Parameter count: {n_params}")

    # Demo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    demo_inference(out_dir, tokenizer, device=device)


if __name__ == "__main__":
    main()
