# Báo cáo tổng kết — ICPR 2026 LRLPR (Low-Resolution License Plate Recognition)

**Tác vụ:** Nhận diện biển số xe từ ảnh độ phân giải thấp, sử dụng 5 frame liên tiếp cho mỗi mẫu, layout Brazil (Brazil-old `LLL-NNNN` hoặc Mercosur `LLL-N-L-NN`).

**Hardware:** RTX 4060 Laptop 8GB VRAM. Mid-range — ảnh hưởng đến lựa chọn kỹ thuật (batch size, model size, training time).

**Mục tiêu:** ≥80% accuracy trên test set 3000 tracks.

---

## 1. Tổng quan kết quả

| Pipeline | Approach | Val acc | Test acc | Δ vs baseline |
|---|---|---|---|---|
| 1 | ResTran baseline | — | 73.00% | — |
| 1 | ResTran + RRDB SR (end-to-end) | — | 73.05% | +0.05 |
| 1 | LP-Diff diffusion SR (offline) | — | — | **ABANDONED** |
| 2 | ICPR2026 V1 (TPS + SE-ResNet34-C + CTC) | 73.10% | 70.37% | -2.63 |
| 2 | ICPR2026 V2 (late fusion + decoder) | 72.50% | 70.63% | -2.37 |
| 2 | ICPR2026 V3 (affine STN, deeper encoder) | 71.60% | 72.23% | -0.77 |
| 2 | ICPR2026 V4 (cross-attn decoder) | 74.80% | 73.87% | +0.87 |
| 2 | ICPR2026 V5 (SVTR + FactTempAttn) | 77.90% | 75.83% | +2.83 |
| 2 | ICPR2026 ensemble 5-way (V1-V5) | 79.20% | 76.93% | +3.93 |
| 3 | Multi-arch SVTR (TPS + SVTR + dual CTC/Attn head) | 78.00% | 77.37% | +4.37 |
| 3 | Multi-arch new_svtr (SVTR 256ch + FactTempAttnNew + SR head) | 78.10% | 76.47% | +3.47 |
| 3 | Multi-arch ResTran (ResNet34 + FactTempAttnNew) | 73.50% | 72.73% | -0.27 |
| 3 | Multi-arch CRNN (CNN + BiLSTM) | 74.20% | 72.17% | -0.83 |
| 3 | **Multi-arch ensemble 4-way (smart vote)** | — | **78.73%** | **+5.73** |
| | Mục tiêu | — | 80.00% | — |

**Submission cuối:** `results/submission_4way_ensemble.txt`

---

## 2. Pipeline 1 — Baseline ResTran + Super-Resolution

### 2.1. ResTranOCR baseline (73.00% test)

**Kiến trúc:**
```
[B, 5, 3, 32, 128]
    → STN (affine, 2x3 theta)
    → ResNet34* (stride layer3/4 = (2,1) để giữ width)
    → AttentionFusion (per-position quality score, weighted sum over 5 frames)
    → Transformer Encoder (3 layers, 8 heads, ff=2048)
    → CTC head (Linear → log_softmax)
```

**File:** `src/models/restran.py`, `src/models/components.py`

**Phân tích lỗi (810 errors):**
- 449 (55%) là **single character swap** ở các cặp khó: M↔H, 6↔8, D↔B, V↔Y, 8↔0/9
- Lengths gần như luôn đúng (chỉ ~5% sai length)
- 60% errors là high-confidence wrong (≥0.9)
- ➜ Bottleneck là **character-level disambiguation**, không phải SR hay length

### 2.2. RRDB end-to-end SR (+0.05 — failed)

**Kiến trúc:** RRDB (8 blocks, nf=32) chèn trước ResTran path. Train joint với CTC loss + L1 SR loss.

**Kết quả:** Chỉ thêm 1 case đúng (+0.05 điểm). Phân tích:
- `sr_feed_hr=False` (default): SR output ×2 rồi downsample về (32,128) → OCR không bao giờ thấy HR
- `sr_feed_hr=True` (test sau đó): OCR cần học lại spatial features → val acc giảm xuống 70.75% và plateau

**Kết luận:** SR không giải được bottleneck (single-char swap ở LR).

### 2.3. LP-Diff (diffusion-based SR) — ABANDONED

**Ý tưởng:** Port LP-Diff (CVPR 2025) — diffusion model chuyên cho biển số. Train standalone, chạy inference 1 lần cache `sr-*.png`, retrain OCR trên SR-preprocessed frames.

**Implementation đầy đủ:**
- `src/models/lpdiff/{mta,unet_diff,diffusion,lpdiff_net}.py` — UNet + Mask Tail Attention + DDPM + DDIM sampler
- `train_lpdiff.py` — standalone trainer với AMP, EMA, val loss, resume

**Kết quả thực tế:** Train 60K iter trên RTX 4060 (bf16, ~3h). Val L1 = 0.0667 nhưng **inference hallucinate ký tự sai**:
- GT `AZD-2548` → SR predict `BVT-7.149`
- GT `AXI-2395` → SR predict `BVT-7.149`

**Nguyên nhân:** Paper gốc dùng 1M iterations. 60K chưa hội tụ. Tăng lên 1M = thêm ~50h training cho diffusion (vẫn chưa kể OCR retrain).

**Kết luận:** Đầu tư thêm 50+h training cho LP-Diff không có giá trị marginal — đổi hướng sang OCR diversity.

---

## 3. Pipeline 2 — ICPR2026 Custom Pipeline (5 variants)

### 3.1. Động lực

Sau khi pipeline 1 không vượt được 73%, chuyển sang đa dạng hoá **OCR architecture**. Đầu tư vào 5 variants khác nhau để ensemble có diversity.

### 3.2. Cấu trúc chung

| Variant | STN | Backbone | Fusion | Decoder | Head | Params |
|---|---|---|---|---|---|---|
| V1 | TPS (10×2=20 pts) | SE-ResNet34-C | QualityFusionMap (early) | — | Linear+CTC | 29M |
| V2 | TPS | SE-ResNet34-C | QualityFusionSeq (late) | CTCDecoder | LN+Linear | 42M |
| V3 | Affine | SE-ResNet34-C | QualityFusionSeq | CTCDecoder | LN+Linear | 48M |
| V4 | Affine | SE-ResNet34-C | stack_frames_as_memory | CrossAttnCTCDecoder | LN+Linear | 54M |
| V5 | TPS | **SVTR (192ch)** | **FactorizedTemporalAttention** | — | Linear+CTC | **8.4M** |

### 3.3. Key modules

**SE-ResNet34-C (`src/models/se_resnet34c.py`):** ResNet34 với Squeeze-and-Excitation block và Bag-of-Tricks "-C" stem (3 conv 3×3 thay 1 conv 7×7).

**STN modes (`src/models/lpr_stn.py`):**
- Affine: 2×3 matrix, identity-initialized
- TPS: 20 control points (10 top + 10 bottom), identity-initialized

**Fusion modules (`src/models/lpr_fusion.py`):**
- `QualityFusionMap` — per-position attention trên feature maps
- `QualityFusionSeq` — per-position attention trên token sequences
- `FactorizedTemporalAttention` — Transformer encoder dọc trục FRAME tại mỗi spatial position

**SVTR backbone (`src/models/svtr.py`):** Vision transformer với patch embed (stride 4), 3 stages (64→128→256 channels), mixed Local (window 7×11) + Global attention, depth (3,6,3).

**Multi-loss (`src/losses/`):**
- MainCTC + AuxCTC (từ intermediate encoder tap)
- CenterLoss trên per-position char features
- OHEM-CTC (hard example mining)
- LengthPenalty (Huber loss vs target_length=7)
- STN regularization (deviation from identity)

### 3.4. Training challenges

**Challenge 1: TPS + AMP fp16 crash trên Windows CUDA**

Symptom: `torch.AcceleratorError: CUDA error: illegal memory access` tại `torch.linalg.solve` trong `_tps_solver` (epoch 4, V1).

Nguyên nhân: fp16 không stable cho linear algebra ops. Trong AMP autocast, mọi op bị ép xuống half-precision → matrix gần singular → CUDA driver crash.

Fix:
1. Wrap `_tps_solver`/`_tps_grid` trong `torch.amp.autocast(enabled=False)`
2. Cast input về float32 explicitly
3. Tikhonov reg `L + 1e-4 * I`
4. V3 (encoder 6 layers) vẫn crash → fallback sang `--stn-mode affine`

Final fix (Pipeline 3): dùng **bf16** thay fp16 → TPS chạy ổn không cần workaround.

**Challenge 2: Loss spike tại epoch ~7**

Symptom: Models V1-V4 train với CTC + AuxCTC + CenterLoss → epoch 1-6 hội tụ đẹp đến ~50-65% val acc → epoch 7 đột ngột: train ctc 0.13→2.78, val 65%→0%, **không recover**.

Nguyên nhân: Khi val acc cross ~65%, model emit fewer blanks → CenterLoss aggregate per-position char features → gradient grow linearly với số non-blank positions. Combined với AuxCTC gradient direction khác → joint optimisation chạm vào vùng brittle.

Fix: **Two-phase training**:
1. Warmup phase: Full composite loss đến ~50% val acc, save checkpoint
2. Fine-tune phase: Resume + `--lambda-center 0 --lambda-aux 0 --lr 5e-5 --grad-clip 0.5` — pure CTC

Recipe này stable, V1 đạt 73.10% val sau 15 epoch fine-tune.

### 3.5. Ensemble 5-way

**Strategy:** Smart vote — mỗi variant cho format-constrained decoded string, voter chọn majority nếu ≥2 agree, else highest-confidence single.

**Path distribution (val):**
- agree_5: 617 (62%) — cả 5 đồng ý
- agree_3-4: 254 (25%)
- conf_X: 129 (13%) — disagree → fallback

**Kết quả: val 79.20%, test 76.93%.**

---

## 4. Pipeline 3 — Multi-Architecture Ensemble (BEST — 78.73% test)

### 4.1. Động lực

Pipeline 2 đạt 76.93% test. Phân tích pairwise agreement giữa V1-V5 cho thấy 85-91% overlap → ensemble không thêm nhiều thông tin. Cần **kiến trúc đa dạng hơn**: Transformer + SVTR + CNN+LSTM mixture.

### 4.2. Khác biệt then chốt vs Pipeline 2

| Khía cạnh | Pipeline 2 (ICPR2026) | Pipeline 3 (Multi-arch) |
|---|---|---|
| AMP precision | fp16 (gặp TPS crash) | **bf16** (stable) |
| TPS implementation | Runtime `linalg.solve` | Pre-computed `inv_delta_C` buffer (no runtime solve) |
| Loss schedule | CTC + AuxCTC + Center + STN reg + (OHEM/Length) | CTC + 0.5·Attention + 0.1·SR (đơn giản, ổn định) |
| Attention decoder | Cross-attention trong V4 (non-AR) | **Autoregressive AttentionDecoder** với teacher forcing trên SVTR variant |
| Backbone diversity | All SE-ResNet34-C (1 SVTR — V5) | 2 SVTR variants + ResNet + CNN-LSTM |

### 4.3. Bốn models trong ensemble

**Model A — SVTR + dual-head (file `src/models/multi_arch/svtr.py`):**
```
TPS (20 pts) → SVTR Backbone (192ch) → TemporalTransformerFusion → Transformer Enc (3L)
    → CTC head + Attention Decoder (autoregressive, teacher forcing)
```
Params: 13M. Test acc: **77.37%** (best single).

**Model B — new SVTR + SR head (file `src/models/multi_arch/new_svtr.py`):**
```
STN (affine) → SVTR Backbone (256ch) → FactorizedTemporalAttention (3 layers, ff=1536)
    → Transformer Enc (4L) → CTC head
    └→ SuperResolutionHead (decode 256ch → HR image, MSE loss với HR ground truth)
```
Params: 14.6M. Test acc: 76.47%. SR head làm regularization phụ.

**Model C — ResTran (file `src/models/multi_arch/restran.py`):**
```
STN (affine) → ResNet34 → FactorizedTemporalAttentionNew → Transformer Enc (3L) → CTC
```
Params: 41.7M. Test acc: 72.73%. Train LR=2e-4 (lower) để tránh spike.

**Model D — CRNN (file `src/models/multi_arch/crnn.py`):**
```
STN (affine) → CNN backbone (7 layers) → AttentionFusion → BiLSTM (2L bidir, hidden=256) → CTC
```
Params: ~9M. Test acc: 72.17%. **Khác biệt kiến trúc nhiều nhất** với 3 model còn lại → tăng diversity ensemble.

### 4.4. Training recipe

**Trainer:** `src/models/multi_arch/trainer.py`

**Loss composition:**
```
L_total = 1.0 * L_ctc + 0.5 * L_attn_ce + 0.1 * L_sr_mse
```
- `L_ctc`: main CTC trên ocr_logits
- `L_attn_ce`: CrossEntropy trên attn_logits (chỉ SVTR variant có)
- `L_sr_mse`: MSE giữa sr_out và HR ground truth (chỉ models có SR head)

**Optimizer/Scheduler:** AdamW + OneCycleLR (max_lr=5e-4, pct_start=0.3 default).

**AMP:** bf16 autocast, GradScaler disabled (bf16 không underflow).

**Per-model training config:**
| Model | Batch size | LR | Epochs | Aux losses |
|---|---|---|---|---|
| SVTR | 48 | 5e-4 | 25 | CTC + Attn |
| new_svtr | 32 | 5e-4 | 25 | CTC + SR |
| ResTran | 16 | 2e-4 | 25 | CTC only (--no-sr) |
| CRNN | 48 | 3e-4 | 25 | CTC only |

### 4.5. Ensemble eval

**File:** `eval_multi_arch.py`

**Strategy:** Smart vote — min_agree=2.
1. Mỗi variant cho format-constrained decoded string + log-score
2. Counter majority — nếu ≥2 agree → return majority string
3. Else → return variant với log-score cao nhất

**Path distribution trên test (3000 samples):**
- agree_4: 1870 (62%) — cả 4 đồng ý
- agree_3: 457 (15%)
- agree_2: 363 (12%)
- conf_X (no majority): 310 (10%) — fallback sang highest-confidence single

**Kết quả: test 78.73%, CER 5.70%.**

---

## 5. Format-Constrained Decoding

**Brazil layouts (mỗi 7 ký tự):**
- Brazil-old: `LLL-NNNN` → `[A-Z]{3}[0-9]{4}`
- Mercosur: `LLL-N-L-NN` → `[A-Z]{3}[0-9][A-Z][0-9]{2}`

**File:** `src/inference/format_decode.py`

**Implementation đã trải qua 3 phiên bản:**

| Version | Strategy | Val acc V1 |
|---|---|---|
| 1 | Chunk CTC sequence thành 7 đoạn đều, argmax per-class | 0.00% |
| 2 | Force pattern fit (project tất cả vào pattern) | 29.20% |
| 3 | **Conditional**: chỉ project khi greedy violate pattern | 73.10% (= greedy) |

**Bài học:** CTC alignment không uniform — chia đều 7 đoạn là sai. Conditional projection neutral nếu greedy đã đúng, chỉ fix khi greedy ra ký tự sai class (chữ ở vị trí số).

Boost thực tế từ format decode: +0.3% trên ensemble (marginal — model đã học implicit pattern).

---

## 6. Engineering Challenges & Lessons

### 6.1. TPS + AMP fp16 → bf16

**Lesson:** Ops dùng linear algebra (`torch.linalg.{solve,lstsq,inv,cholesky}`, SVD, eigendecomp) **không stable với fp16 trên CUDA Windows**.

**Solutions theo thứ tự ưu tiên:**
1. Dùng **bf16** thay fp16 (preferred — bf16 có dynamic range của fp32)
2. Wrap ops trong `autocast(enabled=False)` + cast `.float()`
3. Tikhonov regularization (`L + ε * I`) để tránh near-singular

Code mới (Pipeline 3): default bf16 → không cần workaround.

### 6.2. Composite loss schedule

**Lesson:** Thêm nhiều aux losses (CenterLoss, AuxCTC, OHEM, LengthPenalty) **dễ tạo gradient conflict** khi model đạt mid-accuracy (~60-65%).

**Solutions:**
- Two-phase training (Pipeline 2): warmup composite → fine-tune pure CTC
- Hoặc loss schedule đơn giản hơn (Pipeline 3): CTC + 0.5·Attn + 0.1·SR — không có CenterLoss

CenterLoss đặc biệt nguy hiểm vì magnitude grow theo số non-blank predictions → spike khi accuracy tăng.

### 6.3. Mamba state-space model trên Windows

**Lesson:** Mamba dùng custom CUDA kernel (`selective_scan_cuda.cu`). Build cần Visual Studio C++ + CUDA Toolkit + nvcc — không khả thi trên 4060 laptop trong scope session.

**Workaround:** Thay Mamba bằng **CRNN (BiLSTM)** — vẫn cung cấp kiến trúc thứ 4 khác biệt với 3 Transformer-based còn lại. Diversity tốt cho ensemble.

### 6.4. Format-constrained decoding mong manh

**Lesson:** Force pattern fit > greedy luôn dẫn đến corruption. Chỉ **conditional projection** (project khi greedy violate pattern) là safe.

### 6.5. Backbone choice cho LR images

**Lesson:** SVTR (192-256 channels, mixed Local+Global attention) **vượt ResNet34** (512 channels) cho LR plates:
- ICPR2026 V5 (SVTR 8.4M params): 75.83% test
- ICPR2026 V4 (ResNet 54M params): 73.87% test
- SVTR nhỏ hơn 6.4× nhưng tốt hơn 2 điểm

Lý do: SVTR mixed attention bảo toàn spatial detail ở LR. ResNet pooling sớm phá vỡ character-level features.

---

## 7. Key Findings (sắp xếp theo độ quan trọng)

### Finding #1: SVTR backbone > ResNet34 cho LR plates
V5 (SVTR 8.4M): 75.83% test vs V4 (ResNet 54M): 73.87% — nhỏ hơn 6.4× nhưng tốt hơn 2 điểm.

### Finding #2: bf16 > fp16 cho models dùng linear algebra
fp16 crash TPS `linalg.solve`; bf16 có dynamic range fp32 → stable. Dùng bf16 default cho mọi pipeline có TPS hoặc linear algebra ops.

### Finding #3: Dual CTC + Attention decoder head > CTC alone
Multi-arch SVTR (dual head): 77.37% test vs ICPR2026 V5 (CTC only, cùng SVTR backbone): 75.83% — **+1.54 điểm**. Attention decoder forces sequential char-level reasoning, complementary cho CTC alignment-free.

### Finding #4: Diversity > số lượng models trong ensemble
- ICPR2026 5-way (4 ResNet variants + 1 SVTR, 85-91% pairwise agreement): test 76.93%
- Multi-arch 4-way (2 SVTR + 1 ResNet + 1 CRNN, ~70% pairwise agreement): test 78.73%
- 4 models mixed-arch boost +1.36 vs best single
- 5 models same-family boost +1.10 vs best single

### Finding #5: Single-character substitution là bottleneck, không phải SR
55% errors là 1-char swap ở visually-similar pairs (M↔H, 6↔8, D↔B). Đầu tư vào SR (RRDB, LP-Diff) chỉ +0.05 điểm. Phải giải bằng **character-level disambiguation** (encoder capacity, attention decoder, ensemble).

### Finding #6: Format-constrained decoding cần CONDITIONAL projection
Force fit always → corrupt. Project chỉ khi greedy violate → safe. Boost thực tế +0.3% (marginal).

### Finding #7: Composite loss (CTC + aux + center) unstable
Two-phase training (warmup + pure-CTC fine-tune) ổn định hơn. Hoặc đơn giản hoá loss schedule (CTC + Attn + SR thay vì + Center).

---

## 8. Kết luận

**Mục tiêu 80% test acc: gần đạt nhưng chưa chạm.**

| Pipeline | Test acc | Gap đến 80% |
|---|---|---|
| Baseline RRDB | 73.00% | -7.00 |
| ICPR2026 ensemble 5-way | 76.93% | -3.07 |
| **Multi-arch ensemble 4-way** | **78.73%** | **-1.27** |

**Cải thiện tổng cộng:** +5.73 điểm test acc qua 3 pipelines.

**Để chạm 80% cần ≥1 trong các bước (theo thứ tự cost-effectiveness):**

1. **Train Mamba (model thứ 5)** — yêu cầu setup Visual Studio + CUDA Toolkit. Ước tính +0.5 điểm (do diversity, khác hẳn Transformer/CNN).
2. **Train đủ 60 epoch** thay vì 25 — ngân sách thời gian gấp 2.4×, ước tính +0.5-1 điểm.
3. **TTA (test-time augmentation)** — 4-8 way (horizontal flip + small rotations) — thường +1-2 điểm.
4. **Pre-trained backbone** — SVTR-base trên synthetic plate dataset, fine-tune trên LRLPR — +1-2 điểm.
5. **Beam search decoding** với character n-gram language model — +0.5-1 điểm.

**Recommendation:** Combine #2 + #3 (60 epoch + TTA) khả thi nhất với hardware hiện tại — không cần thêm setup. Ước tính: ~10h training + 2h inference TTA → 80%+ là khả thi.

---

## 9. Reproducibility

Để reproduce kết quả 78.73% test (~4h trên RTX 4060):

```bash
# Verify environment
python -c "import torch; print(torch.cuda.is_available())"

# Train 4 models (mỗi model ~50 min)
python train_multi_arch.py -n multi_svtr     -m svtr     --epochs 25 --batch-size 48 --lr 5e-4
python train_multi_arch.py -n multi_new_svtr -m new_svtr --epochs 25 --batch-size 32 --lr 5e-4
python train_multi_arch.py -n multi_restran  -m restran  --epochs 25 --batch-size 16 --lr 2e-4 --no-sr
python train_multi_arch.py -n multi_crnn     -m crnn     --epochs 25 --batch-size 48 --lr 3e-4

# Evaluate ensemble on labelled test set
python eval_multi_arch.py \
  --ckpt svtr=results/multi_svtr_best.pth \
  --ckpt new_svtr=results/multi_new_svtr_best.pth \
  --ckpt restran=results/multi_restran_best.pth \
  --ckpt crnn=results/multi_crnn_best.pth \
  --min-agree 2 --mode test_labeled

# Generate competition submission (unlabeled test mode)
python eval_multi_arch.py \
  --ckpt svtr=results/multi_svtr_best.pth \
  --ckpt new_svtr=results/multi_new_svtr_best.pth \
  --ckpt restran=results/multi_restran_best.pth \
  --ckpt crnn=results/multi_crnn_best.pth \
  --min-agree 2 --mode test --output results/submission_final.txt
```

---

## 10. Project structure

```
MultiFrame-LPR/
├── README.md                              # Quick start
├── REPORT.md                              # File này
├── pyproject.toml, uv.lock                # Dependencies
│
├── configs/
│   ├── config.py                          # Pipeline 1 (legacy ResTran) config
│   ├── icpr2026_base.py                   # Pipeline 2 (V1-V5) base
│   ├── icpr2026_variants.py               # Pipeline 2 per-variant overrides
│   └── lpdiff_config.py                   # LP-Diff (abandoned)
│
├── src/
│   ├── data/                              # MultiFrameDataset + augmentation
│   ├── models/
│   │   ├── components.py                  # Pipeline 1: legacy modules
│   │   ├── crnn.py, restran.py, sr_model.py # Pipeline 1: baselines + RRDB
│   │   ├── lpdiff/                        # LP-Diff (abandoned)
│   │   ├── se_resnet34c.py                # Pipeline 2: SE-ResNet34-C
│   │   ├── svtr.py                        # Pipeline 2: SVTR backbone
│   │   ├── lpr_stn.py                     # Pipeline 2: Affine + TPS rectifier
│   │   ├── lpr_fusion.py                  # Pipeline 2: Quality + FactTempAttn fusion
│   │   ├── lpr_encoder.py lpr_decoder.py  # Pipeline 2: encoder/decoder modules
│   │   ├── lpr_variants.py                # Pipeline 2: V1-V5 assembly
│   │   └── multi_arch/                    # Pipeline 3 (BEST)
│   │       ├── components.py              #   TPS + SVTR + FactTempAttn + AttnDec + SR
│   │       ├── svtr.py new_svtr.py        #   SVTROCR + svtrNew
│   │       ├── restran.py crnn.py mamba.py
│   │       └── trainer.py                 #   UniversalTrainer (bf16, dual-head loss)
│   ├── losses/                            # Pipeline 2: multi-loss modules
│   ├── inference/                         # Ensemble + format-constrained decoding
│   ├── training/
│   │   ├── trainer.py                     # Pipeline 1: ResTran trainer
│   │   └── icpr2026_trainer.py            # Pipeline 2: V1-V5 trainer
│   └── utils/
│
├── train.py                               # Pipeline 1 entrypoint
├── train_icpr2026.py                      # Pipeline 2 entrypoint
├── train_multi_arch.py                    # Pipeline 3 entrypoint (BEST)
├── train_lpdiff.py                        # LP-Diff (abandoned)
│
├── eval_icpr2026.py                       # Pipeline 2 per-variant eval
├── eval_icpr2026_ensemble_v2.py           # Pipeline 2 ensemble eval
├── eval_multi_arch.py                     # Pipeline 3 ensemble eval (BEST)
│
├── results/
│   ├── icpr2026_v{1-5}_best.pth           # Pipeline 2 checkpoints
│   ├── multi_{svtr,new_svtr,restran,crnn}_best.pth  # Pipeline 3 checkpoints
│   ├── restran_*best.pth                  # Pipeline 1 checkpoints
│   └── submission_4way_ensemble.txt       # Final submission (78.73% test)
│
├── archive/                               # Experimental/legacy scripts
└── logs/                                  # Training logs
```

---

## 11. References (chỉ kiến trúc, không phải sources implementation)

- **SVTR** — Du et al. 2022 "SVTR: Scene Text Recognition with a Single Visual Model" (kiến trúc backbone)
- **Four-stage STR framework** — Baek et al. 2019 "What Is Wrong With Scene Text Recognition Model Comparisons?" (kiến trúc tổng thể)
- **TPS-STN** — Shi et al. 2016 "Robust Scene Text Recognition with Automatic Rectification"
- **CTC** — Graves et al. 2006
- **SE block** — Hu et al. 2018 "Squeeze-and-Excitation Networks"
- **Bag-of-Tricks ResNet-C** — He et al. 2019
- **RRDB** — Wang et al. 2018 ESRGAN
- **Mamba SSM** — Gu & Dao 2023 (cài đặt fail trên Windows)
- **LP-Diff** — Gong et al. CVPR 2025 (abandoned do training cost)
