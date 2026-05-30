"""Dataset utilities for multi-frame license plate recognition."""

import glob
import json
import os
import random
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.transforms import (
    get_degradation_transforms,
    get_hr_transforms,
    get_light_transforms,
    get_train_transforms,
    get_val_transforms,
)


class MultiFrameDataset(Dataset):
    """Dataset for the released ICPR 2026 LRLPR train/test layout.

    Expected layout:
        train/Scenario-A/<layout>/track_xxxxx/
        train/Scenario-B/<layout>/track_xxxxx/
        test/track_xxxxx/

    ``train`` mode uses the train portion of a reproducible train/val split.
    ``val`` mode uses the validation portion of that split.
    ``test`` mode indexes every labelled test track without splitting.
    ``is_test=True`` keeps the old unlabeled submission behavior.
    """

    REQUIRED_FRAMES = 5

    def __init__(
        self,
        root_dir: str,
        mode: str = "train",
        split_ratio: float = 0.9,
        img_height: int = 32,
        img_width: int = 128,
        char2idx: Dict[str, int] | None = None,
        val_split_file: str = "data/LRLPR-26-5opEvJTW/val_tracks.json",
        seed: int = 42,
        augmentation_level: str = "full",
        is_test: bool = False,
        full_train: bool = False,
        load_hr: bool = False,
        hr_height: int = 64,
        hr_width: int = 256,
        use_sr_cache: bool = False,
        sr_prefix: str = "sr",
    ):
        """
        Args:
            root_dir: Directory containing track folders, recursively.
            mode: ``train``, ``val``, or ``test``.
            split_ratio: Fraction of labelled train tracks used for training.
            img_height: Target image height.
            img_width: Target image width.
            char2idx: Character to index mapping.
            val_split_file: JSON file storing validation track relative paths.
            seed: Random seed for reproducible splitting.
            augmentation_level: ``full`` or ``light`` training augmentation.
            is_test: If True, load tracks without requiring annotations.
            full_train: If True, use all labelled tracks for training.
            load_hr: If True, load optional ``hr-*.png`` frames for SR loss.
            hr_height: Target HR tensor height.
            hr_width: Target HR tensor width.
            use_sr_cache: If True, prefer ``sr_prefix``-prefixed files (e.g.
                ``sr-001.png``) over the raw ``lr-*`` frames. Used after running
                ``gen_sr_images.py`` to feed OCR on LP-Diff outputs.
            sr_prefix: Filename prefix for the cached SR frames. Default "sr".
        """
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported dataset mode: {mode}")

        self.root_dir = os.path.abspath(root_dir)
        self.mode = mode
        self.samples: List[Dict[str, Any]] = []
        self.img_height = img_height
        self.img_width = img_width
        self.char2idx = char2idx or {}
        self.val_split_file = val_split_file
        self.seed = seed
        self.augmentation_level = augmentation_level
        self.is_test = is_test
        self.full_train = full_train
        self.load_hr = load_hr
        self.hr_height = hr_height
        self.hr_width = hr_width
        self.hr_transform = get_hr_transforms(hr_height, hr_width) if load_hr else None
        self.use_sr_cache = use_sr_cache
        self.sr_prefix = sr_prefix

        if mode == "train":
            self.transform = (
                get_light_transforms(img_height, img_width)
                if augmentation_level == "light"
                else get_train_transforms(img_height, img_width)
            )
            self.degrade = get_degradation_transforms()
        else:
            self.transform = get_val_transforms(img_height, img_width)
            self.degrade = None

        print(f"[{mode.upper()}] Scanning: {root_dir}")
        all_tracks = self._find_tracks()
        if not all_tracks:
            print("ERROR: No track folders found.")
            return

        if is_test:
            print(f"[TEST] Loaded {len(all_tracks)} unlabeled tracks.")
            self._index_unlabeled_samples(all_tracks)
            print(f"-> Total: {len(self.samples)} test samples.")
            return

        if mode == "test":
            print(f"[TEST] Loaded {len(all_tracks)} labelled tracks.")
            self._index_labelled_samples(all_tracks, include_synthetic=False)
            print(f"-> Total: {len(self.samples)} test samples.")
            return

        train_tracks, val_tracks = self._load_or_create_split(all_tracks, split_ratio)
        selected_tracks = train_tracks if mode == "train" else val_tracks
        print(f"[{mode.upper()}] Loaded {len(selected_tracks)} tracks.")
        self._index_labelled_samples(
            selected_tracks,
            include_synthetic=(mode == "train"),
        )
        print(f"-> Total: {len(self.samples)} samples.")

    def _find_tracks(self) -> List[str]:
        """Return all track directories under ``root_dir``."""
        search_path = os.path.join(self.root_dir, "**", "track_*")
        return sorted(path for path in glob.glob(search_path, recursive=True) if os.path.isdir(path))

    def _track_key(self, track_path: str) -> str:
        """Stable relative id used for split files and logs."""
        rel_path = os.path.relpath(track_path, self.root_dir)
        return rel_path.replace(os.sep, "/")

    def _load_or_create_split(
        self,
        all_tracks: List[str],
        split_ratio: float,
    ) -> Tuple[List[str], List[str]]:
        """Load an existing train/val split or create a random one."""
        if self.full_train:
            print("FULL TRAIN MODE: using all labelled tracks for training.")
            return all_tracks, []

        if not 0.0 < split_ratio < 1.0:
            raise ValueError(f"split_ratio must be in (0, 1), got {split_ratio}")

        track_by_key = {self._track_key(track): track for track in all_tracks}
        track_by_basename: Dict[str, List[str]] = {}
        for track in all_tracks:
            track_by_basename.setdefault(os.path.basename(track), []).append(track)

        val_tracks: List[str] = []
        if os.path.exists(self.val_split_file):
            print(f"Loading split from '{self.val_split_file}'...")
            try:
                with open(self.val_split_file, "r", encoding="utf-8") as f:
                    raw_ids = json.load(f)
                val_ids = [str(item).replace("\\", "/") for item in raw_ids]
            except Exception as exc:
                print(f"WARNING: Could not read split file ({exc}). Recreating split.")
                val_ids = []

            # New files store relative paths. Old files stored only basename ids.
            if val_ids and any("/" in item for item in val_ids):
                val_tracks = [track_by_key[item] for item in val_ids if item in track_by_key]
            elif val_ids:
                for item in val_ids:
                    val_tracks.extend(track_by_basename.get(item, []))

            if val_tracks:
                val_set = set(val_tracks)
                train_tracks = [track for track in all_tracks if track not in val_set]
                return train_tracks, val_tracks

            print("WARNING: Split file did not match current dataset. Recreating split.")

        shuffled_tracks = list(all_tracks)
        random.Random(self.seed).shuffle(shuffled_tracks)
        val_size = max(1, int(round(len(shuffled_tracks) * (1.0 - split_ratio))))
        val_tracks = sorted(shuffled_tracks[:val_size])
        val_set = set(val_tracks)
        train_tracks = [track for track in all_tracks if track not in val_set]

        split_dir = os.path.dirname(self.val_split_file)
        if split_dir:
            os.makedirs(split_dir, exist_ok=True)
        with open(self.val_split_file, "w", encoding="utf-8") as f:
            json.dump([self._track_key(track) for track in val_tracks], f, indent=2)
        print(f"Saved validation split with {len(val_tracks)} tracks to {self.val_split_file}.")

        return train_tracks, val_tracks

    @staticmethod
    def _read_label(track_path: str) -> str:
        """Read plate text from a track annotation file."""
        json_path = os.path.join(track_path, "annotations.json")
        if not os.path.exists(json_path):
            return ""

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return ""

        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict):
            return ""

        label = data.get("plate_text") or data.get("license_plate") or data.get("text") or ""
        return str(label).strip().upper()

    @staticmethod
    def _image_files(track_path: str, prefix: str) -> List[str]:
        """Return sorted png/jpg/jpeg frame files for a prefix."""
        files: List[str] = []
        for ext in ("png", "jpg", "jpeg"):
            files.extend(glob.glob(os.path.join(track_path, f"{prefix}-*.{ext}")))
        return sorted(files)

    def _index_labelled_samples(self, tracks: List[str], include_synthetic: bool) -> None:
        """Index labelled LR samples, optionally adding degraded HR samples."""
        skipped_without_label = 0
        skipped_without_frames = 0

        sr_used = 0
        sr_missing = 0
        for track_path in tqdm(tracks, desc=f"Indexing {self.mode}"):
            label = self._read_label(track_path)
            if not label:
                skipped_without_label += 1
                continue

            track_id = self._track_key(track_path)
            lr_files = self._image_files(track_path, "lr")
            hr_files = self._image_files(track_path, "hr")
            if self.use_sr_cache:
                sr_files = self._image_files(track_path, self.sr_prefix)
                if sr_files:
                    lr_files = sr_files
                    sr_used += 1
                else:
                    sr_missing += 1
            if lr_files:
                self.samples.append(
                    {
                        "paths": lr_files,
                        "hr_paths": hr_files,
                        "label": label,
                        "is_synthetic": False,
                        "track_id": track_id,
                    }
                )
            else:
                skipped_without_frames += 1

            if include_synthetic and hr_files:
                self.samples.append(
                    {
                        "paths": hr_files,
                        "hr_paths": hr_files,
                        "label": label,
                        "is_synthetic": True,
                        "track_id": track_id,
                    }
                )

        if skipped_without_label:
            print(f"WARNING: skipped {skipped_without_label} tracks without labels.")
        if skipped_without_frames:
            print(f"WARNING: skipped {skipped_without_frames} labelled tracks without LR frames.")
        if self.use_sr_cache:
            print(f"SR cache: used '{self.sr_prefix}-*' on {sr_used} tracks; "
                  f"fell back to 'lr-*' on {sr_missing} tracks.")

    def _index_unlabeled_samples(self, tracks: List[str]) -> None:
        """Index tracks for submission/inference when labels are unavailable."""
        for track_path in tqdm(tracks, desc="Indexing test"):
            lr_files = self._image_files(track_path, "lr")
            if self.use_sr_cache:
                sr_files = self._image_files(track_path, self.sr_prefix)
                if sr_files:
                    lr_files = sr_files
            if not lr_files:
                continue
            self.samples.append(
                {
                    "paths": lr_files,
                    "hr_paths": self._image_files(track_path, "hr"),
                    "label": "",
                    "is_synthetic": False,
                    "track_id": self._track_key(track_path),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize_frame_count(self, img_paths: List[str]) -> List[str]:
        """Pad or truncate a track to the fixed frame count expected by models."""
        if len(img_paths) >= self.REQUIRED_FRAMES:
            return img_paths[: self.REQUIRED_FRAMES]
        return img_paths + [img_paths[-1]] * (self.REQUIRED_FRAMES - len(img_paths))

    def _load_hr_tensor(self, hr_paths: List[str]) -> Tuple[torch.Tensor, bool]:
        """Load clean HR frames as an SR target tensor [5, 3, hr_h, hr_w].

        Returns (hr_tensor, has_hr). If HR files are missing or loading is disabled
        a zeros tensor with the correct shape is returned and has_hr=False.
        """
        zeros = torch.zeros(
            self.REQUIRED_FRAMES, 3, self.hr_height, self.hr_width,
            dtype=torch.float32,
        )
        if not self.load_hr or self.hr_transform is None or not hr_paths:
            return zeros, False

        paths = self._normalize_frame_count(hr_paths)
        images: List[np.ndarray] = []
        max_h, max_w = 0, 0
        for path in paths:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                return zeros, False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            max_h, max_w = max(max_h, h), max(max_w, w)
            images.append(image)

        padded: List[np.ndarray] = []
        for image in images:
            h, w = image.shape[:2]
            if h != max_h or w != max_w:
                image = cv2.copyMakeBorder(
                    image, 0, max_h - h, 0, max_w - w, cv2.BORDER_REPLICATE,
                )
            padded.append(image)

        transformed = self.hr_transform(
            image=padded[0],
            image1=padded[1],
            image2=padded[2],
            image3=padded[3],
            image4=padded[4],
        )
        hr_tensor = torch.stack(
            [
                transformed["image"],
                transformed["image1"],
                transformed["image2"],
                transformed["image3"],
                transformed["image4"],
            ],
            dim=0,
        )
        return hr_tensor, True

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, str, str, torch.Tensor, bool]:
        item = self.samples[idx]
        img_paths = self._normalize_frame_count(item["paths"])
        label = item["label"]
        is_synthetic = item["is_synthetic"]
        track_id = item["track_id"]

        images_list = []
        max_h, max_w = 0, 0
        for path in img_paths:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if is_synthetic and self.degrade:
                image = self.degrade(image=image)["image"]

            h, w = image.shape[:2]
            max_h, max_w = max(max_h, h), max(max_w, w)
            images_list.append(image)

        padded_images = []
        for image in images_list:
            h, w = image.shape[:2]
            if h != max_h or w != max_w:
                image = cv2.copyMakeBorder(
                    image,
                    0,
                    max_h - h,
                    0,
                    max_w - w,
                    cv2.BORDER_REPLICATE,
                )
            padded_images.append(image)

        transformed = self.transform(
            image=padded_images[0],
            image1=padded_images[1],
            image2=padded_images[2],
            image3=padded_images[3],
            image4=padded_images[4],
        )
        images_tensor = torch.stack(
            [
                transformed["image"],
                transformed["image1"],
                transformed["image2"],
                transformed["image3"],
                transformed["image4"],
            ],
            dim=0,
        )

        if self.is_test:
            target = [0]
        else:
            target = [self.char2idx[char] for char in label if char in self.char2idx]
            if not target:
                target = [0]

        hr_tensor, has_hr = self._load_hr_tensor(item.get("hr_paths", []))

        return (
            images_tensor,
            torch.tensor(target, dtype=torch.long),
            len(target),
            label,
            track_id,
            hr_tensor,
            has_hr,
        )

    @staticmethod
    def collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor, int, str, str, torch.Tensor, bool]]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[str, ...],
        Tuple[str, ...],
        torch.Tensor,
        torch.Tensor,
    ]:
        """Custom collate function for variable-length CTC targets and optional HR."""
        (
            images,
            targets,
            target_lengths,
            labels_text,
            track_ids,
            hr_frames,
            has_hr_flags,
        ) = zip(*batch)
        return (
            torch.stack(images, 0),
            torch.cat(targets),
            torch.tensor(target_lengths, dtype=torch.long),
            labels_text,
            track_ids,
            torch.stack(hr_frames, 0),
            torch.tensor(has_hr_flags, dtype=torch.bool),
        )
