"""
Konfigürasyon parametreleri.
Türkçe açıklamalar ve basit dict tabanlı kullanım.
"""

CONFIG = {
    # Model seçim: tercihen "google/flan-t5-small" (seq2seq) veya "distilgpt2" (causal)
    "model_name": "google/flan-t5-small",

    # Veri seti tercih (ilk denenilecek): databricks/dolly-15k
    "dataset_name": "databricks/dolly-15k",
    # Alternatif: "tatsu-lab/alpaca"

    # Veri sınırlama
    "max_samples": 50000,

    # Tokenizer / model giriş uzunluğu
    "max_length": 512,

    # Eğitim parametreleri
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "fp16": True,
    "warmup_ratio": 0.05,

    # Çıktı klasörü
    "output_dir": "./outputs/slm-model",

    # Rastgelelik
    "seed": 42,

    # Kayıt ve log ayarları (Trainer argümanlarıyla uyumlu küçük set)
    "logging_steps": 50,
    "save_steps": 500,
    "eval_steps": 500,
}

def get_config():
    return CONFIG
