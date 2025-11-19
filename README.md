# TurkStudent CO - LLM Jetpack Ders Notları

Bu depo, TurkStudent CO tarafından hazırlanan "LLM Jetpack" dersine ait ders notlarını ve örnek Jupyter notebook'ları içerir. Ders materyalleri eğitim amaçlıdır ve model eğitimi, ince ayar (fine-tuning) ve örnek kullanım senaryolarını kapsar.

## İçindekiler (kısa)

- `basıt_llm.ipynb` — Basit BERT tabanlı sınıflandırma örneği (IMDb veri seti ile). Model eğitimi ve değerlendirme adımları içerir.
- `minnak_llm.ipynb` — Küçük ölçekli LLM fine-tuning örneği; SmolLM tabanlı ince ayar ve test kodları. Bellek optimizasyonu örnekleri de mevcuttur.
- `model_kıyaslama.ipynb` — Farklı modellerin (T5, BERT, GPT-2) karşılaştırmalı örnekleri: özetleme, soru-cevap ve metin üretimi.
- `veri_çekme.ipynb` — Veri indirme ve ön işleme örnekleri.
- `Nlp_1.ipynb`, `Transformers_.ipynb`, `Metrikler.ipynb`, `Kantizasyon.ipynb`, `FineTunne.ipynb` — Dersin farklı bölümlerinde kullanılan yardımcı notebook'lar ve deneyler.

## Kurulum (hızlı)

1. Python 3.10 veya 3.11 kullanmanızı öneririz.
2. Sanal ortam oluşturun ve etkinleştirin:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Gerekli paketleri yükleyin:

```bash
pip install -r requirements.txt
```

> Not: Eğer GPU kullanıyorsanız, `torch` paketini sisteminizin CUDA sürümüne göre kurun (ör. https://pytorch.org/get-started/locally/).

## Çalıştırma (örnek)

1. İlgili notebook'u JupyterLab veya Jupyter Notebook ile açın:

```bash
jupyter lab
```

2. Notebooks içinde hücreleri sırayla çalıştırın. Büyük modeller çalıştırmadan önce `minnak_llm.ipynb` içerisindeki `MODEL_NAME`, batch/epoch ve dataset boyutu gibi parametreleri kontrol edin.

