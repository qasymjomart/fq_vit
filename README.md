# FQ-ViT Reproduction (ViT-B / CIFAR-100)

> A focused reproduction of **[FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer](https://github.com/megvii-research/FQ-ViT)**, restricted to the **ViT-B** backbone, evaluated on **CIFAR-100**.

---

## 📌 Overview

This repository reproduces the FQ-ViT post-training quantization pipeline on a single backbone — **ViT-B** — to validate the method's accuracy retention claims on a smaller-scale dataset than the original ImageNet benchmark.

- 🧠 **Backbone:** ViT-B (fine-tuned on CIFAR-100 prior to quantization)
- 📦 **Dataset:** CIFAR-100
- ⚙️ **Method:** FQ-ViT post-training quantization (`QAct`, `IntLayerNorm`, `QIntSoftmax`)
- 🎯 **Scope:** Single-model reproduction (no multi-architecture sweep)

---

## 🛠️ Setup

1. ViT-B was first **fine-tuned on CIFAR-100** to obtain a full-precision baseline checkpoint.
2. The fine-tuned checkpoint was then passed through the **FQ-ViT** post-training quantization pipeline.
3. Both the full-precision and quantized models were evaluated on the CIFAR-100 test/validation split using standard Top-1 / Top-5 accuracy metrics.

---

## 📊 Results

| Model              | Prec@1      | Prec@5      |
|---------------------|:-----------:|:-----------:|
| 🟢 Full Precision    | **91.270**  | **98.540**  |
| 🔵 Quantized (FQ-ViT) | **90.210**  | **98.230**  |

**Δ Accuracy Drop:** `Prec@1: −1.060` · `Prec@5: −0.310`

> ✅ The quantized model retains the vast majority of full-precision accuracy, consistent with the trends reported in the original FQ-ViT paper, despite the shift in dataset scale (ImageNet → CIFAR-100) and the single-backbone scope of this reproduction.

---

## ⚠️ Scope & Limitations

- Only **ViT-B** was reproduced; other backbones from the original paper (e.g., DeiT, Swin variants) were **not** tested.
- Evaluation was performed on **CIFAR-100**, not the original ImageNet benchmark — results are not directly comparable to the paper's reported numbers.
- ViT-B was fine-tuned on CIFAR-100 prior to quantization, rather than quantizing an ImageNet-pretrained checkpoint directly.

---

## 🙏 Credits

This work is a reproduction built directly on top of the original FQ-ViT research:

- **Paper:** Lin, Y., et al. *"FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer."* IJCAI 2022.
- **Original Repository:** [megvii-research/FQ-ViT](https://github.com/megvii-research/FQ-ViT)

All credit for the original method, architecture design, and core implementation belongs to the original authors. This repository only adapts and evaluates their approach on a different backbone/dataset configuration for reproduction purposes.

---

## 📄 License

This reproduction follows the licensing terms of the original FQ-ViT repository. Please refer to the [original repo](https://github.com/megvii-research/FQ-ViT) for full license details.