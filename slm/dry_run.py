"""
dry_run.py

Hızlı doğrulama için: `kids_examples.md` içeriğini alır, küçük bir dataset oluşturur
ve çok küçük bir model ile 1 epoch'luk eğitim çalıştırır. Amaç pipeline'ı test etmektir.

Not: Bu dry-run gerçekteki tam eğitim ile farklıdır (küçük model, az veri, 1 epoch).
"""

import os
import logging

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
from datasets import Dataset

from dataset import clean_examples, tokenize_unified

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_kids_examples(path):
    """Parse kids_examples.md which is in the unified format and return dict lists."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    parts = text.split("### Instruction:")
    ins = []
    outs = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if "### Response:" in p:
            i, r = p.split("### Response:", 1)
            ins.append(i.strip())
            outs.append(r.strip())
    return {"instruction": ins, "response": outs}


def expand_dataset(unified, target_size=80):
    # Repeat examples to reach target_size (simple augmentation for dry-run)
    ins, outs = unified["instruction"], unified["response"]
    cur = len(ins)
    if cur == 0:
        raise ValueError("No kid examples found")
    while len(ins) < target_size:
        ins.extend(ins[: min(cur, target_size - len(ins))])
        outs.extend(outs[: min(cur, target_size - len(outs))])
    return {"instruction": ins[:target_size], "response": outs[:target_size]}


def main():
    base = os.path.dirname(__file__)
    kids_path = os.path.join(base, "kids_examples.md")
    logger.info(f"Loading kids examples from {kids_path}")
    unified = load_kids_examples(kids_path)
    unified = expand_dataset(unified, target_size=80)
    unified = clean_examples(unified)
    logger.info(f"Prepared {len(unified['instruction'])} cleaned examples for dry-run")

    # Try a small public model; fallback if not available
    candidate_models = ["sshleifer/tiny-t5", "t5-small", "google/flan-t5-small"]
    tokenizer = None
    model_name = None
    for m in candidate_models:
        try:
            logger.info(f"Trying tokenizer/model: {m}")
            tokenizer = AutoTokenizer.from_pretrained(m)
            model_name = m
            break
        except Exception as e:
            logger.warning(f"Model {m} not available locally or via HF: {e}")

    if tokenizer is None:
        raise RuntimeError("No suitable model found for dry-run; please check network or tokens")

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    tokenized = tokenize_unified(unified, tokenizer, max_length=128)
    ds = Dataset.from_dict(tokenized)
    split = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    logger.info(f"Train/Eval sizes: {len(train_ds)}/{len(eval_ds)}")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    args = Seq2SeqTrainingArguments(
        output_dir="./outputs/dry_run",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="no",
        fp16=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting dry-run training (1 epoch)")
    trainer.train()
    logger.info("Dry-run finished. Running inference on 3 samples:")

    samples = unified["instruction"][:3]
    for s in samples:
        prompt = f"### Instruction:\n{s}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(trainer.model.device) for k, v in inputs.items()}
        out = trainer.model.generate(**inputs, max_new_tokens=64)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print("\nPROMPT:", s)
        print("OUTPUT:", text)


if __name__ == "__main__":
    main()
