"""Generate a notebook for visualizing SR/RRDB checkpoint behavior."""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


NOTEBOOK_PATH = Path("notebooks/visualize_sr_checkpoints.ipynb")


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}

nb.cells = [
    md(
        """
        # SR / RRDB Checkpoint Visualizer

        Notebook nay dung de inspect tung track trong test set:

        - Hien thi 5 frame LR goc.
        - Hien thi RRDB/SR output 2x.
        - Hien thi anh SR sau khi downsample ve input size cua OCR.
        - So sanh prediction, ground truth, confidence cua nhieu checkpoint.
        - Tu dong chon ca case dung va sai, va luon include `track_10119`.

        Cac checkpoint duoc so sanh gom baseline no-SR, SR v4 joint, SR v7 feed-HR,
        SR fix e1, va checkpoint ghep RRDB v4 + OCR baseline.
        """
    ),
    code(
        """
        from pathlib import Path
        import os
        import sys
        import glob
        import json
        import random
        from collections import OrderedDict

        import cv2
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn.functional as F
        import matplotlib.pyplot as plt
        from IPython.display import display

        ROOT = Path.cwd()
        if not (ROOT / "configs").exists():
            ROOT = ROOT.parent
        os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))

        from configs.config import Config
        from src.data.transforms import get_val_transforms
        from src.models.restran import ResTranOCR
        from src.utils.common import seed_everything
        from src.utils.postprocess import decode_with_confidence

        seed_everything(42)
        cfg = Config()
        NUM_FRAMES = getattr(cfg, "NUM_FRAMES", 5)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("ROOT:", ROOT)
        print("DEVICE:", DEVICE)
        print("NUM_FRAMES:", NUM_FRAMES)
        """
    ),
    md(
        """
        ## 1. Checkpoint registry

        `sr_v4_rrdb_plus_baseline_73.03` la checkpoint tot nhat hien tai:

        - RRDB weights lay tu `restran_sr_v4_best.pth`
        - OCR weights lay tu baseline `restran_best.pth`
        - Test accuracy da do: `73.03%`

        `sr_v7_feedhr_joint_62.23` la checkpoint dang mo trong tab cua ban,
        test accuracy da do: `62.23%`.
        """
    ),
    code(
        """
        CHECKPOINTS = OrderedDict({
            "baseline_no_sr_73.00": {
                "path": "results/restran_best.pth",
                "csv": "results/test_predictions_restran.csv",
                "use_sr": False,
                "sr_feed_hr": False,
                "sr_blend": 0.0,
                "note": "ResTranOCR baseline, no SR",
            },
            "sr_v4_rrdb_plus_baseline_73.03": {
                "path": "results/restran_baseline_ocr_v4_sr_best.pth",
                "csv": "results/test_predictions_restran_baseline_ocr_v4_sr_unclamped.csv",
                "use_sr": True,
                "sr_feed_hr": False,
                "sr_blend": 1.0,
                "note": "RRDB from v4 + OCR baseline; best measured SR pipeline",
            },
            "sr_v4_joint_72.47": {
                "path": "results/restran_sr_v4_best.pth",
                "csv": "results/test_predictions_restran_sr_v4_cpu.csv",
                "use_sr": True,
                "sr_feed_hr": False,
                "sr_blend": 1.0,
                "note": "Joint-trained RRDB + OCR from v4",
            },
            "sr_v7_feedhr_joint_62.23": {
                "path": "results/restran_sr_v7_best.pth",
                "csv": "results/test_predictions_restran_sr_v7_rerun.csv",
                "use_sr": True,
                "sr_feed_hr": True,
                "sr_blend": 1.0,
                "note": "v7 feed-HR joint checkpoint; low test accuracy",
            },
            "sr_fix_e1_72.57": {
                "path": "results/restran_sr_fix_e1_best.pth",
                "csv": "results/test_predictions_restran_sr_fix_e1.csv",
                "use_sr": True,
                "sr_feed_hr": False,
                "sr_blend": 1.0,
                "note": "1 epoch SR-only fine-tune from merged checkpoint",
            },
        })

        SR_MODEL_NAMES = [name for name, spec in CHECKPOINTS.items() if spec["use_sr"]]

        for name, spec in CHECKPOINTS.items():
            print(f"{name:36s}", Path(spec["path"]).exists(), spec["note"])
        """
    ),
    md("## 2. Read prediction CSVs and pick representative test tracks"),
    code(
        """
        def read_pred_csv(path):
            path = Path(path)
            if not path.exists():
                return pd.DataFrame(
                    columns=["track_id", "prediction", "confidence", "ground_truth", "correct"]
                )
            df = pd.read_csv(path)
            df["track_id"] = df["track_id"].astype(str)
            df["prediction"] = df["prediction"].astype(str)
            df["ground_truth"] = df["ground_truth"].astype(str)
            df["correct"] = df["correct"].astype(int)
            return df


        pred_tables = {name: read_pred_csv(spec["csv"]) for name, spec in CHECKPOINTS.items()}

        summary = []
        for name, df in pred_tables.items():
            correct = int(df["correct"].sum()) if len(df) else None
            total = len(df) if len(df) else None
            acc = 100.0 * correct / total if total else None
            summary.append((name, correct, total, acc, CHECKPOINTS[name]["note"]))

        summary_df = pd.DataFrame(
            summary,
            columns=["checkpoint", "correct", "total", "acc_percent", "note"],
        )
        display(summary_df)
        """
    ),
    code(
        """
        base = pred_tables["baseline_no_sr_73.00"].set_index("track_id")
        best = pred_tables["sr_v4_rrdb_plus_baseline_73.03"].set_index("track_id")
        v7 = pred_tables["sr_v7_feedhr_joint_62.23"].set_index("track_id")

        all_ids = sorted(set(base.index) & set(best.index) & set(v7.index))
        cases = pd.DataFrame({
            "track_id": all_ids,
            "gt": [base.loc[t, "ground_truth"] for t in all_ids],
            "base_pred": [base.loc[t, "prediction"] for t in all_ids],
            "base_ok": [int(base.loc[t, "correct"]) for t in all_ids],
            "best_sr_pred": [best.loc[t, "prediction"] for t in all_ids],
            "best_sr_ok": [int(best.loc[t, "correct"]) for t in all_ids],
            "v7_pred": [v7.loc[t, "prediction"] for t in all_ids],
            "v7_ok": [int(v7.loc[t, "correct"]) for t in all_ids],
        })


        def take_ids(mask, n, seed=42):
            pool = cases.loc[mask, "track_id"].tolist()
            rng = random.Random(seed)
            rng.shuffle(pool)
            return pool[:n]


        selected_ids = []
        selected_ids += ["track_10119"]
        selected_ids += take_ids((cases.base_ok == 0) & (cases.best_sr_ok == 1), 4, seed=1)
        selected_ids += take_ids((cases.base_ok == 1) & (cases.best_sr_ok == 0), 4, seed=2)
        selected_ids += take_ids((cases.v7_ok == 0) & (cases.best_sr_ok == 1), 4, seed=3)
        selected_ids += take_ids(
            (cases.base_ok == 1) & (cases.best_sr_ok == 1) & (cases.v7_ok == 1),
            4,
            seed=4,
        )
        selected_ids += take_ids(
            (cases.base_ok == 0) & (cases.best_sr_ok == 0) & (cases.v7_ok == 0),
            4,
            seed=5,
        )
        selected_ids = list(OrderedDict.fromkeys([t for t in selected_ids if t in all_ids]))


        def case_bucket(row):
            if row.base_ok == 0 and row.best_sr_ok == 1:
                return "best_sr_fixes_baseline"
            if row.base_ok == 1 and row.best_sr_ok == 0:
                return "best_sr_breaks_baseline"
            if row.v7_ok == 0 and row.best_sr_ok == 1:
                return "v7_fails_best_sr_ok"
            if row.base_ok == 1 and row.best_sr_ok == 1 and row.v7_ok == 1:
                return "all_correct"
            if row.base_ok == 0 and row.best_sr_ok == 0 and row.v7_ok == 0:
                return "all_wrong"
            return "mixed"


        cases["bucket"] = cases.apply(case_bucket, axis=1)
        display(cases[cases.track_id.isin(selected_ids)])
        print("Selected tracks:", selected_ids)
        """
    ),
    md("## 3. Model loading and image helpers"),
    code(
        """
        MEAN = torch.tensor([0.485, 0.456, 0.406])
        STD = torch.tensor([0.229, 0.224, 0.225])
        VAL_TRANSFORM = get_val_transforms(cfg.IMG_HEIGHT, cfg.IMG_WIDTH)
        model_cache = {}


        def build_model(spec):
            model = ResTranOCR(
                num_classes=cfg.NUM_CLASSES,
                transformer_heads=cfg.TRANSFORMER_HEADS,
                transformer_layers=cfg.TRANSFORMER_LAYERS,
                transformer_ff_dim=cfg.TRANSFORMER_FF_DIM,
                dropout=cfg.TRANSFORMER_DROPOUT,
                use_stn=cfg.USE_STN,
                pretrained=False,
                use_sr=spec["use_sr"],
                sr_num_blocks=getattr(cfg, "SR_NUM_BLOCKS", 8),
                sr_scale=getattr(cfg, "SR_SCALE", 2),
                sr_nf=getattr(cfg, "SR_NF", 32),
                sr_gc=getattr(cfg, "SR_GC", 16),
                sr_feed_hr=spec["sr_feed_hr"],
                sr_blend=spec["sr_blend"],
                sr_use_checkpoint=False,
            )
            return model.to(DEVICE)


        def load_state_dict(path):
            state = torch.load(path, map_location=DEVICE)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            return state


        def get_model(name):
            if name not in model_cache:
                spec = CHECKPOINTS[name]
                model = build_model(spec)
                state = load_state_dict(spec["path"])
                missing, unexpected = model.load_state_dict(state, strict=False)
                model.eval()
                model_cache[name] = model
                print(f"loaded {name}: missing={len(missing)}, unexpected={len(unexpected)}")
            return model_cache[name]


        def track_path(track_id):
            direct = Path(cfg.TEST_DATA_ROOT) / track_id
            if direct.exists():
                return direct
            matches = list(Path(cfg.TEST_DATA_ROOT).rglob(track_id))
            if not matches:
                raise FileNotFoundError(track_id)
            return matches[0]


        def read_label(track_dir):
            with open(Path(track_dir) / "annotations.json", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                data = data[0] if data else {}
            return str(data.get("plate_text") or data.get("license_plate") or data.get("text") or "").strip().upper()


        def image_files(track_dir, prefix="lr"):
            files = []
            for ext in ("png", "jpg", "jpeg"):
                files.extend(glob.glob(str(Path(track_dir) / f"{prefix}-*.{ext}")))
            files = sorted(files)
            if not files:
                raise FileNotFoundError(f"No {prefix}-*.png/jpg/jpeg files in {track_dir}")
            if len(files) >= NUM_FRAMES:
                return files[:NUM_FRAMES]
            return files + [files[-1]] * (NUM_FRAMES - len(files))


        def load_raw_frames(track_id, prefix="lr"):
            files = image_files(track_path(track_id), prefix=prefix)
            frames = []
            for file in files:
                img = cv2.imread(file, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Could not read {file}")
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return frames


        def pad_frames_like_dataset(frames):
            max_h = max(img.shape[0] for img in frames)
            max_w = max(img.shape[1] for img in frames)
            padded = []
            for img in frames:
                h, w = img.shape[:2]
                if h != max_h or w != max_w:
                    img = cv2.copyMakeBorder(
                        img,
                        0,
                        max_h - h,
                        0,
                        max_w - w,
                        cv2.BORDER_REPLICATE,
                    )
                padded.append(img)
            return padded


        def preprocess_frames(frames):
            padded = pad_frames_like_dataset(frames)
            transformed = VAL_TRANSFORM(
                image=padded[0],
                image1=padded[1],
                image2=padded[2],
                image3=padded[3],
                image4=padded[4],
            )
            x = torch.stack(
                [
                    transformed["image"],
                    transformed["image1"],
                    transformed["image2"],
                    transformed["image3"],
                    transformed["image4"],
                ],
                dim=0,
            ).unsqueeze(0)
            resized_lr = [cv2.resize(img, (cfg.IMG_WIDTH, cfg.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR) for img in padded]
            return x, resized_lr


        def tensor_to_rgb(x):
            x = x.detach().float().cpu().permute(1, 2, 0).numpy()
            return np.clip(x, 0.0, 1.0)


        def normalized_tensor_to_rgb(x):
            x = x.detach().float().cpu()
            x = x * STD.view(3, 1, 1) + MEAN.view(3, 1, 1)
            return tensor_to_rgb(x)
        """
    ),
    code(
        """
        def predict_one(model_name, x):
            model = get_model(model_name)
            with torch.no_grad():
                logits = model(x.to(DEVICE))
            pred, conf = decode_with_confidence(logits, cfg.IDX2CHAR)[0]
            return pred, float(conf)


        def rrdb_visuals(model_name, x):
            model = get_model(model_name)
            if not getattr(model, "use_sr", False):
                return None
            with torch.no_grad():
                _, sr = model(x.to(DEVICE), return_sr=True)
            sr_lr = F.interpolate(sr, size=(cfg.IMG_HEIGHT, cfg.IMG_WIDTH), mode="bilinear", align_corners=False)
            return {
                "sr_hr": [tensor_to_rgb(sr[i]) for i in range(sr.shape[0])],
                "sr_down_for_ocr": [tensor_to_rgb(sr_lr[i]) for i in range(sr_lr.shape[0])],
            }
        """
    ),
    md("## 4. Prediction and visualization functions"),
    code(
        """
        def prediction_table(track_id):
            rows = []
            for name, spec in CHECKPOINTS.items():
                df = pred_tables.get(name, pd.DataFrame())
                if len(df) and track_id in set(df["track_id"]):
                    r = df[df["track_id"] == track_id].iloc[0]
                    rows.append({
                        "checkpoint": name,
                        "prediction_csv": r["prediction"],
                        "confidence_csv": float(r["confidence"]),
                        "ground_truth": r["ground_truth"],
                        "correct_csv": int(r["correct"]),
                        "note": spec["note"],
                    })
                else:
                    rows.append({
                        "checkpoint": name,
                        "prediction_csv": None,
                        "confidence_csv": None,
                        "ground_truth": read_label(track_path(track_id)),
                        "correct_csv": None,
                        "note": spec["note"],
                    })
            return pd.DataFrame(rows)


        def run_live_predictions(track_id):
            frames = load_raw_frames(track_id)
            x, _ = preprocess_frames(frames)
            gt = read_label(track_path(track_id))
            rows = []
            for name, spec in CHECKPOINTS.items():
                pred, conf = predict_one(name, x)
                rows.append({
                    "checkpoint": name,
                    "live_prediction": pred,
                    "live_confidence": conf,
                    "ground_truth": gt,
                    "live_correct": int(pred == gt),
                    "note": spec["note"],
                })
            return pd.DataFrame(rows)


        def show_frame_strip(images, title, cmap=None, figsize=(16, 2.6)):
            fig, axes = plt.subplots(1, len(images), figsize=figsize)
            fig.suptitle(title, y=1.08, fontsize=13)
            if len(images) == 1:
                axes = [axes]
            for i, (ax, img) in enumerate(zip(axes, images), start=1):
                ax.imshow(img, cmap=cmap)
                ax.set_title(f"frame {i}")
                ax.axis("off")
            plt.show()


        def show_diff_strip(a_images, b_images, title, figsize=(16, 2.6)):
            diffs = []
            for a, b in zip(a_images, b_images):
                a = cv2.resize(a, (b.shape[1], b.shape[0])).astype(np.float32) / 255.0
                b = b.astype(np.float32)
                diffs.append(np.abs(b - a).mean(axis=2))
            show_frame_strip(diffs, title, cmap="magma", figsize=figsize)


        def show_model_grid(rows_by_model, title, figsize=None):
            n_rows = len(rows_by_model)
            n_cols = NUM_FRAMES
            if figsize is None:
                figsize = (17, max(2.6, 2.5 * n_rows))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            if n_rows == 1:
                axes = np.expand_dims(axes, 0)
            fig.suptitle(title, y=1.01, fontsize=13)
            for r, (name, images) in enumerate(rows_by_model.items()):
                for c, img in enumerate(images):
                    axes[r, c].imshow(img)
                    axes[r, c].set_title(f"{name}\\nf{c + 1}", fontsize=9)
                    axes[r, c].axis("off")
            fig.tight_layout()
            plt.show()
        """
    ),
    code(
        """
        def visualize_rrdb_grid(track_id, model_names=None):
            model_names = model_names or SR_MODEL_NAMES
            raw = load_raw_frames(track_id)
            x, resized_lr = preprocess_frames(raw)

            hr_rows = OrderedDict()
            down_rows = OrderedDict()
            for name in model_names:
                visuals = rrdb_visuals(name, x)
                if visuals is None:
                    continue
                hr_rows[name] = visuals["sr_hr"]
                down_rows[name] = visuals["sr_down_for_ocr"]

            show_frame_strip(resized_lr, f"{track_id}: LR resized to OCR input 32x128", figsize=(16, 2.3))
            show_model_grid(hr_rows, f"{track_id}: RRDB output HR 64x256 for each SR checkpoint")
            show_model_grid(down_rows, f"{track_id}: SR downsampled to 32x128 for OCR")


        def visualize_track(track_id, sr_model_name="sr_v4_rrdb_plus_baseline_73.03", show_all_rrdb=False):
            print("=" * 120)
            print("TRACK:", track_id)
            print("GT:", read_label(track_path(track_id)))

            display(prediction_table(track_id))
            live = run_live_predictions(track_id)
            display(live)

            raw = load_raw_frames(track_id)
            x, resized_lr = preprocess_frames(raw)

            show_frame_strip(raw, "Original raw LR frames from test track", figsize=(16, 2.3))
            show_frame_strip(resized_lr, "LR frames resized to OCR input 32x128", figsize=(16, 2.3))

            visuals = rrdb_visuals(sr_model_name, x)
            if visuals is not None:
                show_frame_strip(visuals["sr_hr"], f"{sr_model_name}: RRDB output HR 64x256", figsize=(16, 3.2))
                show_frame_strip(visuals["sr_down_for_ocr"], f"{sr_model_name}: SR downsampled back to 32x128 for OCR", figsize=(16, 2.3))
                show_diff_strip(resized_lr, visuals["sr_down_for_ocr"], f"{sr_model_name}: mean absolute visual difference vs resized LR")

            if show_all_rrdb:
                visualize_rrdb_grid(track_id)
        """
    ),
    md(
        """
        ## 5. Visualize selected correct/wrong cases

        Cell nay visualize nhieu case. Neu GPU memory thieu, giam `selected_ids[:16]`
        xuong `selected_ids[:5]`, hoac restart kernel roi chay tung `visualize_track("track_xxxxx")`.
        """
    ),
    code(
        """
        print("Selected tracks:", selected_ids)
        for track_id in selected_ids[:16]:
            visualize_track(track_id)
        """
    ),
    md("## 6. Inspect one specific track and all RRDB checkpoints"),
    code(
        """
        # Track ban dang chon trong CSV v7:
        visualize_track("track_10119", show_all_rrdb=True)
        """
    ),
    code(
        """
        # Chi so sanh anh SR/RRDB cua tat ca checkpoint, khong lap lai raw/prediction table.
        visualize_rrdb_grid("track_10119")
        """
    ),
    md(
        """
        ## 7. Optional: save visual panels to disk

        Cell nay luu panel cua tung track vao `results/visual_debug/`.
        Moi panel gom LR + HR RRDB + SR-downsample cua tat ca SR checkpoint.
        """
    ),
    code(
        """
        SAVE_DIR = Path("results/visual_debug")
        SAVE_DIR.mkdir(parents=True, exist_ok=True)


        def save_contact_sheet(track_id, sr_model_names=None):
            sr_model_names = sr_model_names or SR_MODEL_NAMES
            raw = load_raw_frames(track_id)
            x, resized_lr = preprocess_frames(raw)
            gt = read_label(track_path(track_id))
            pred_df = prediction_table(track_id)

            rows = OrderedDict()
            rows["LR 32x128"] = resized_lr
            for name in sr_model_names:
                visuals = rrdb_visuals(name, x)
                if visuals is None:
                    continue
                rows[f"{name} HR"] = [(img * 255).astype(np.uint8) for img in visuals["sr_hr"]]
                rows[f"{name} down"] = [(img * 255).astype(np.uint8) for img in visuals["sr_down_for_ocr"]]

            n_rows = len(rows)
            fig, axes = plt.subplots(n_rows, NUM_FRAMES, figsize=(18, 2.8 * n_rows))
            if n_rows == 1:
                axes = np.expand_dims(axes, 0)
            fig.suptitle(f"{track_id} | GT={gt}", fontsize=14)
            for r, (label, row_imgs) in enumerate(rows.items()):
                for c, img in enumerate(row_imgs):
                    axes[r, c].imshow(img)
                    axes[r, c].set_title(f"{label}\\nf{c + 1}", fontsize=8)
                    axes[r, c].axis("off")
            fig.tight_layout()
            out = SAVE_DIR / f"{track_id}.png"
            fig.savefig(out, dpi=160, bbox_inches="tight")
            plt.close(fig)

            pred_df.to_csv(SAVE_DIR / f"{track_id}_predictions.csv", index=False)
            return out


        saved = [save_contact_sheet(t) for t in selected_ids[:16]]
        saved
        """
    ),
]

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH}")
