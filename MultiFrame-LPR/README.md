# MultiFrame-LPR

Multi-frame OCR solution cho **ICPR 2026 Low-Resolution License Plate Recognition Challenge**. Pipeline xử lý 5 frame liên tiếp cho mỗi biển số, kết hợp SVTR/ResNet backbones, factorized temporal attention, dual CTC/Attention decoder heads, và ensemble 4 kiến trúc khác nhau.

🔗 [Challenge homepage](https://icpr26lrlpr.github.io/)

📄 **Báo cáo chi tiết:** xem [REPORT.md](REPORT.md) — tổng kết toàn bộ các kỹ thuật đã thử nghiệm, so sánh kết quả, và các bài học engineering.

---

## Kết quả

| Pipeline | Test acc | Δ vs baseline |
|---|---|---|
| 1 — ResTran + RRDB SR | 73.00% | — |
| 2 — ICPR2026 ensemble 5-way (V1-V5) | 76.93% | +3.93 |
| **3 — Multi-architecture ensemble 4-way** ⭐ | **78.73%** | **+5.73** |
| Mục tiêu | 80.00% | |

**File submission cuối cùng:** `results/submission_4way_ensemble.txt`

---

## Quick Start (reproduce 78.73% test acc)

```bash
# Install dependencies
uv sync

# Data structure: data/LRLPR-26-5opEvJTW/{train/Scenario-{A,B}, test}/

# 1. Train 4 multi-arch models (~4h total trên RTX 4060 8GB)
python train_multi_arch.py -n multi_svtr     -m svtr     --epochs 25 --batch-size 48 --lr 5e-4
python train_multi_arch.py -n multi_new_svtr -m new_svtr --epochs 25 --batch-size 32 --lr 5e-4
python train_multi_arch.py -n multi_restran  -m restran  --epochs 25 --batch-size 16 --lr 2e-4 --no-sr
python train_multi_arch.py -n multi_crnn     -m crnn     --epochs 25 --batch-size 48 --lr 3e-4

# 2. Ensemble eval trên labelled test set
python eval_multi_arch.py \
  --ckpt svtr=results/multi_svtr_best.pth \
  --ckpt new_svtr=results/multi_new_svtr_best.pth \
  --ckpt restran=results/multi_restran_best.pth \
  --ckpt crnn=results/multi_crnn_best.pth \
  --min-agree 2 --mode test_labeled

# 3. Generate competition submission
python eval_multi_arch.py \
  --ckpt svtr=results/multi_svtr_best.pth \
  --ckpt new_svtr=results/multi_new_svtr_best.pth \
  --ckpt restran=results/multi_restran_best.pth \
  --ckpt crnn=results/multi_crnn_best.pth \
  --min-agree 2 --mode test --output results/submission_final.txt
```

---

## Pipelines available

Project chứa **3 pipelines song song**, mỗi pipeline có entrypoint riêng.

### Pipeline 3 — Multi-architecture ensemble (BEST — 78.73% test)

Ensemble 4 kiến trúc khác nhau, mỗi model có training recipe + loss schedule khác nhau, gộp bằng smart majority voting.

```bash
python train_multi_arch.py -m {svtr,new_svtr,restran,crnn} -n my_exp [--epochs 25 ...]
python eval_multi_arch.py --ckpt v1=... --mode {val,test_labeled,test}
```

4 architectures:
- **svtr**: TPS + SVTR backbone + TemporalFusion + **dual CTC/Attention head** (best single: 77.37% test)
- **new_svtr**: STN + SVTR (256ch) + FactorizedTempAttn + SR head (76.47% test)
- **restran**: STN + ResNet34 + FactorizedTempAttn (72.73% test)
- **crnn**: STN + CNN + BiLSTM — kiến trúc khác biệt nhất → tăng diversity ensemble (72.17% test)

Files: `src/models/multi_arch/`, `train_multi_arch.py`, `eval_multi_arch.py`

### Pipeline 2 — ICPR2026 custom (V1-V5 — 76.93% test ensemble)

5 variants với SE-ResNet34-C backbone, khác nhau ở vị trí multi-frame fusion và decoder.

```bash
python train_icpr2026.py --variant {v1,v2,v3,v4,v5} [--epochs 25 ...]
python eval_icpr2026.py --variant v1 --ckpt results/icpr2026_v1_best.pth --eval-test-labeled
python eval_icpr2026_ensemble_v2.py --ckpt v1=... --ckpt v2=... --mode test_labeled
```

Files: `src/models/{se_resnet34c,svtr,lpr_*}.py`, `src/models/lpr_variants.py`

### Pipeline 1 — Legacy ResTran + RRDB SR (baseline — 73.00% test)

Pipeline ban đầu với ResNet34 + AttentionFusion + RRDB super-resolution.

```bash
python train.py --model {restran,crnn} [-n my_exp]
```

Files: `src/models/{restran,crnn,components,sr_model}.py`

---

## Dataset format

```
data/LRLPR-26-5opEvJTW/
├── train/
│   ├── Scenario-A/
│   │   └── <layout>/track_*/lr-*.{png,jpg}  (+ hr-*.png + annotations.json)
│   └── Scenario-B/...
└── test/track_*/lr-*.{png,jpg}  (+ annotations.json)
```

**annotations.json:** `{"plate_text": "ABC1234"}` (7 ký tự, Brazil-old hoặc Mercosur).

---

## Project structure

```
configs/                  Config dataclasses (3 pipelines)
src/
├── data/                 MultiFrameDataset + augmentation
├── models/
│   ├── components.py restran.py crnn.py sr_model.py  Pipeline 1
│   ├── lpdiff/                                       LP-Diff (abandoned)
│   ├── se_resnet34c.py svtr.py lpr_*.py              Pipeline 2 modules
│   ├── lpr_variants.py                               Pipeline 2: V1-V5 assembly
│   └── multi_arch/                                   Pipeline 3 (BEST)
│       ├── components.py svtr.py new_svtr.py restran.py crnn.py mamba.py
│       └── trainer.py
├── losses/               Pipeline 2 multi-loss modules
├── inference/            Ensemble + format-constrained decoding
├── training/             Trainers cho Pipeline 1 + 2
└── utils/

train.py                  Pipeline 1
train_icpr2026.py         Pipeline 2
train_multi_arch.py       Pipeline 3 (BEST)
train_lpdiff.py           LP-Diff (abandoned)

eval_icpr2026.py                Per-variant eval cho Pipeline 2
eval_icpr2026_ensemble_v2.py    Pipeline 2 ensemble
eval_multi_arch.py              Pipeline 3 ensemble (BEST)
run_ensemble.py                 Legacy ensemble (Pipeline 1 variants)

archive/                  Experimental scripts
logs/                     Training logs
results/                  Checkpoints + submissions
REPORT.md                 Comprehensive technical report
```

---

## Hardware

Đã verify trên **RTX 4060 Laptop 8GB VRAM** (Windows 11). Batch sizes đã tinh chỉnh để fit. CPU-only inference work nhưng chậm (~50× slower).

---

## Documentation

- **[REPORT.md](REPORT.md)** — Báo cáo tổng kết: kiến trúc, thí nghiệm, so sánh kết quả 3 pipelines, bài học engineering
