"""
dataset.py

Yükleme, temizleme ve tokenizasyon için yardımcı fonksiyonlar.

Kullanım:
  from dataset import prepare_datasets
  train_ds, eval_ds = prepare_datasets(cfg, tokenizer)

Bu modül HuggingFace Datasets API'sini kullanır.
"""

from datasets import load_dataset
import logging
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def _guess_fields(example):
    # Örnek sözlükte ortak alan adlarını kontrol et
    keys = set(example.keys())
    if "instruction" in keys and "response" in keys:
        return "instruction", "response"
    if "instruction" in keys and "output" in keys:
        return "instruction", "output"
    if "prompt" in keys and "completion" in keys:
        return "prompt", "completion"
    if "input" in keys and "output" in keys and "instruction" in keys:
        return "instruction", "output"
    # Fallback: tek metin alanı olabilir
    if "text" in keys:
        return "text", None
    # yoksa None
    return None, None


def _make_unified(example, in_field, out_field):
    # Birleştirilmiş formatı oluştur
    if in_field is None and out_field is None:
        return None
    instruction = example.get(in_field, "") if in_field else ""
    response = example.get(out_field, "") if out_field else ""
    # Çıktı yoksa bazı datasetlerde response text prompt'tan ayrılır
    return {
        "instruction": instruction if instruction is not None else "",
        "response": response if response is not None else "",
    }


def load_and_convert(dataset_name, max_samples=50000):
    """
    HuggingFace üzerinden dataset yükler, örnekleri unified formatına çevirir
    ve en fazla `max_samples` kadarını döner.
    """
    logger.info(f"Loading dataset {dataset_name}")
    ds = load_dataset(dataset_name)
    # dataset yapısı farklı olabilir; önce en büyük split'i al
    # tercih: 'train' split varsa onu kullan
    split_key = "train" if "train" in ds.keys() else list(ds.keys())[0]
    records = ds[split_key]

    # Limitleme
    total = len(records)
    take = min(total, max_samples)
    if take < total:
        logger.info(f"Selecting first {take} of {total} samples")
        records = records.select(range(take))

    unified = {"instruction": [], "response": []}
    # try to guess fields using first example
    if len(records) == 0:
        raise ValueError("Dataset contains no examples")

    in_field, out_field = _guess_fields(records[0])
    logger.info(f"Guessed fields: in={in_field}, out={out_field}")

    for ex in records:
        if in_field is None and out_field is None:
            # skip
            continue
        u = _make_unified(ex, in_field, out_field)
        if u is None:
            continue
        unified["instruction"].append(u["instruction"])
        unified["response"].append(u["response"])

    logger.info(f"Converted to unified format: {len(unified['instruction'])} examples")
    return unified


def clean_examples(unified, min_len=5):
    """
    Basit temizleme: boş/çok kısa örnekleri çıkar.
    """
    assert "instruction" in unified and "response" in unified
    ins, outs = unified["instruction"], unified["response"]
    kept_in, kept_out = [], []
    for i, (a, b) in enumerate(zip(ins, outs)):
        if a is None or b is None:
            continue
        a_stripped = str(a).strip()
        b_stripped = str(b).strip()
        if len(a_stripped) < min_len:
            continue
        if len(b_stripped) < min_len:
            continue
        kept_in.append(a_stripped)
        kept_out.append(b_stripped)
    logger.info(f"Cleaned: {len(kept_in)} kept from {len(ins)}")
    return {"instruction": kept_in, "response": kept_out}


def make_prompt(instruction, response=None):
    # Unified textual prompt per talimat
    prompt = """
### Instruction:
{instruction}

### Response:
""".strip()
    return prompt.format(instruction=instruction)


def tokenize_unified(unified, tokenizer: PreTrainedTokenizerBase, max_length=512):
    """
    Tokenize input (prompt) and labels (response). Return dict list ready for Trainer.
    """
    inputs = []
    targets = []
    for ins, out in zip(unified["instruction"], unified["response"]):
        src = make_prompt(ins)
        inputs.append(src)
        targets.append(out)

    # Tokenize inputs
    model_inputs = tokenizer(inputs, truncation=True, max_length=max_length)
    # Tokenize labels separately
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, truncation=True, max_length=max_length)

    # Replace pad token id's in labels by -100 to ignore in loss
    label_ids = labels["input_ids"]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
    for i in range(len(label_ids)):
        label_ids[i] = [tok if tok != pad_id else -100 for tok in label_ids[i]]

    model_inputs["labels"] = label_ids
    return model_inputs


def prepare_datasets(cfg, tokenizer):
    """
    Yükleme, temizleme ve tokenizasyonu birleştirir. Dönen veri Trainer ile kullanılabilir dict formatındadır.
    """
    raw = load_and_convert(cfg["dataset_name"], cfg["max_samples"]) if cfg.get("dataset_name") else {}
    cleaned = clean_examples(raw)
    tokenized = tokenize_unified(cleaned, tokenizer, max_length=cfg.get("max_length", 512))
    # Dönüş: model_inputs as dict of lists; kullanıcı HuggingFace Dataset.from_dict ile dönüştürebilir
    return tokenized
