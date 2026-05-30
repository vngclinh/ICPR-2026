"""Dataset for LP-Diff training and inference.

Returns three LR frames upscaled to HR resolution plus the HR target, all in
[-1, 1] (diffusion convention). Reuses MultiFrameDataset's track discovery and
train/val split logic via a thin proxy so the LP-Diff split exactly matches the
OCR split.
"""
from __future__ import annotations

import glob
import json
import os
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


REQUIRED_FRAMES = 5


def _track_key(track_path: str, root_dir: str) -> str:
    rel = os.path.relpath(track_path, root_dir)
    return rel.replace(os.sep, "/")


def _image_files(track_path: str, prefix: str) -> List[str]:
    files: List[str] = []
    for ext in ("png", "jpg", "jpeg"):
        files.extend(glob.glob(os.path.join(track_path, f"{prefix}-*.{ext}")))
    return sorted(files)


def _find_tracks(root_dir: str) -> List[str]:
    search_path = os.path.join(root_dir, "**", "track_*")
    return sorted(
        path for path in glob.glob(search_path, recursive=True) if os.path.isdir(path)
    )


def _load_or_create_split(
    all_tracks: List[str], root_dir: str, val_split_file: str,
    split_ratio: float, seed: int,
) -> Tuple[List[str], List[str]]:
    """Reuse the OCR validation split so SR training never sees the val tracks."""
    track_by_key = {_track_key(t, root_dir): t for t in all_tracks}
    track_by_basename: dict[str, list[str]] = {}
    for t in all_tracks:
        track_by_basename.setdefault(os.path.basename(t), []).append(t)

    val_tracks: List[str] = []
    if os.path.exists(val_split_file):
        with open(val_split_file, "r", encoding="utf-8") as f:
            raw_ids = json.load(f)
        val_ids = [str(item).replace("\\", "/") for item in raw_ids]
        if val_ids and any("/" in item for item in val_ids):
            val_tracks = [track_by_key[k] for k in val_ids if k in track_by_key]
        elif val_ids:
            for k in val_ids:
                val_tracks.extend(track_by_basename.get(k, []))
        if val_tracks:
            val_set = set(val_tracks)
            return [t for t in all_tracks if t not in val_set], val_tracks

    shuffled = list(all_tracks)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(round(len(shuffled) * (1.0 - split_ratio))))
    val_tracks = sorted(shuffled[:val_size])
    val_set = set(val_tracks)
    return [t for t in all_tracks if t not in val_set], val_tracks


def _read_image(path: str) -> Optional[np.ndarray]:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _resize_to(image: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


def _to_minus1_1(image_uint8: np.ndarray) -> torch.Tensor:
    """HWC uint8 -> CHW float32 in [-1, 1]."""
    t = torch.from_numpy(image_uint8).permute(2, 0, 1).float() / 255.0
    return t * 2.0 - 1.0


class LPDiffDataset(Dataset):
    """Returns dict(LR1, LR2, LR3, HR, track_id) tensors in [-1, 1].

    Args:
        root_dir: Train/test directory containing track folders (recursively).
        mode: "train", "val", or "all" (all tracks under root_dir).
        hr_height/hr_width: Resolution that BOTH the LR frames (bicubic up) and
            the HR target are resized to.
        frame_indices: Which 3 of the 5 frames to feed the MTA module.
        require_hr: When True, skip tracks that have no hr-*.png (HR is needed
            for training). Set to False for inference-only datasets.
        val_split_file: Reuse the OCR train/val split JSON.
    """

    def __init__(
        self,
        root_dir: str,
        mode: str = "train",
        hr_height: int = 64,
        hr_width: int = 256,
        frame_indices: Tuple[int, int, int] = (0, 2, 4),
        require_hr: bool = True,
        val_split_file: str = "data/LRLPR-26-5opEvJTW/val_tracks.json",
        split_ratio: float = 0.9,
        seed: int = 42,
    ):
        super().__init__()
        if mode not in {"train", "val", "all"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.root_dir = os.path.abspath(root_dir)
        self.mode = mode
        self.hr_height = hr_height
        self.hr_width = hr_width
        self.frame_indices = frame_indices
        self.require_hr = require_hr

        all_tracks = _find_tracks(self.root_dir)
        if mode == "all":
            selected = all_tracks
        else:
            train_tracks, val_tracks = _load_or_create_split(
                all_tracks, self.root_dir, val_split_file, split_ratio, seed,
            )
            selected = train_tracks if mode == "train" else val_tracks

        self.tracks: List[dict] = []
        for track_path in tqdm(selected, desc=f"LP-Diff[{mode}] indexing"):
            lr_files = _image_files(track_path, "lr")
            hr_files = _image_files(track_path, "hr")
            if not lr_files:
                continue
            if self.require_hr and not hr_files:
                continue
            self.tracks.append({
                "lr_paths": lr_files,
                "hr_paths": hr_files,
                "track_id": _track_key(track_path, self.root_dir),
                "track_path": track_path,
            })
        print(f"LP-Diff[{mode}]: {len(self.tracks)} tracks usable.")

    def __len__(self) -> int:
        return len(self.tracks)

    def _pick_frame(self, paths: List[str], idx: int) -> str:
        n = len(paths)
        return paths[idx] if idx < n else paths[-1]

    def __getitem__(self, idx: int) -> dict:
        item = self.tracks[idx]
        lr_paths = item["lr_paths"]
        hr_paths = item["hr_paths"]

        i1, i2, i3 = self.frame_indices
        lr_imgs: List[torch.Tensor] = []
        for fi in (i1, i2, i3):
            path = self._pick_frame(lr_paths, fi)
            img = _read_image(path)
            if img is None:
                img = np.zeros((self.hr_height, self.hr_width, 3), dtype=np.uint8)
            img = _resize_to(img, self.hr_height, self.hr_width)
            lr_imgs.append(_to_minus1_1(img))

        if hr_paths:
            hr_img = _read_image(self._pick_frame(hr_paths, 0))
            if hr_img is None:
                hr_img = np.zeros((self.hr_height, self.hr_width, 3), dtype=np.uint8)
        else:
            hr_img = np.zeros((self.hr_height, self.hr_width, 3), dtype=np.uint8)
        hr_img = _resize_to(hr_img, self.hr_height, self.hr_width)

        return {
            "LR1": lr_imgs[0],
            "LR2": lr_imgs[1],
            "LR3": lr_imgs[2],
            "HR": _to_minus1_1(hr_img),
            "track_id": item["track_id"],
            "track_path": item["track_path"],
            "has_hr": bool(hr_paths),
        }


def lpdiff_collate(batch: List[dict]) -> dict:
    """Stack tensor fields; keep string/bool fields as tuples."""
    out: dict = {}
    keys = batch[0].keys()
    for k in keys:
        values = [b[k] for b in batch]
        if isinstance(values[0], torch.Tensor):
            out[k] = torch.stack(values, dim=0)
        else:
            out[k] = values
    return out
