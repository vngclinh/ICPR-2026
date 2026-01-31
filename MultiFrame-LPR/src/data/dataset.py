"""MultiFrameDataset for license plate recognition with multi-frame input."""
import glob
import json
import os
import random
from typing import Any, Dict, List, Tuple

import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_degradation_transforms,
    get_light_transforms,
)


class MultiFrameDataset(Dataset):
    """Dataset for multi-frame license plate recognition.
    
    Handles both real LR images and synthetic LR (degraded HR) images.
    Implements Scenario-B specific validation splitting logic.
    """
    
    def __init__(
        self,
        root_dir: str,
        mode: str = 'train',
        split_ratio: float = 0.9,
        img_height: int = 32,
        img_width: int = 128,
        char2idx: Dict[str, int] = None,
        val_split_file: str = "data/val_tracks.json",
        seed: int = 42,
        augmentation_level: str = "full",
        is_test: bool = False,
        full_train: bool = False,
    ):
        """
        Args:
            root_dir: Root directory containing track folders.
            mode: 'train' or 'val'.
            split_ratio: Train/val split ratio.
            img_height: Target image height.
            img_width: Target image width.
            char2idx: Character to index mapping.
            val_split_file: Path to validation split JSON file.
            seed: Random seed for reproducible splitting.
            augmentation_level: 'full' or 'light' augmentation for training.
            is_test: If True, load test data without labels (for submission).
            full_train: If True, use all tracks for training (no val split).
        """
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
        
        if mode == 'train':
            # Training: apply augmentation on the fly
            if augmentation_level == "light":
                self.transform = get_light_transforms(img_height, img_width)
            else:
                self.transform = get_train_transforms(img_height, img_width)
            self.degrade = get_degradation_transforms()
        else:
            # Validation or test: only resize and normalize
            self.transform = get_val_transforms(img_height, img_width)
            self.degrade = None

        print(f"[{mode.upper()}] Scanning: {root_dir}")
        abs_root = os.path.abspath(root_dir)
        search_path = os.path.join(abs_root, "**", "track_*")
        all_tracks = sorted(glob.glob(search_path, recursive=True))
        
        if not all_tracks:
            print("❌ ERROR: No data found.")
            return

        # Handle test mode differently
        if is_test:
            print(f"[TEST] Loaded {len(all_tracks)} tracks.")
            self._index_test_samples(all_tracks)
            print(f"-> Total: {len(self.samples)} test samples.")
        else:
            train_tracks, val_tracks = self._load_or_create_split(all_tracks, split_ratio)
            
            selected_tracks = train_tracks if mode == 'train' else val_tracks
            print(f"[{mode.upper()}] Loaded {len(selected_tracks)} tracks.")
            
            self._index_samples(selected_tracks)
            print(f"-> Total: {len(self.samples)} samples.")

    def _load_or_create_split(
        self,
        all_tracks: List[str],
        split_ratio: float
    ) -> Tuple[List[str], List[str]]:
        """Load existing split or create new one with Scenario-B priority."""
        # If full_train mode, return all tracks as training
        if self.full_train:
            print("📌 FULL TRAIN MODE: Using all tracks for training (no validation split).")
            return all_tracks, []
        
        train_tracks, val_tracks = [], []
        
        # 1. Load split file if exists
        if os.path.exists(self.val_split_file):
            print(f"📂 Loading split from '{self.val_split_file}'...")
            try:
                with open(self.val_split_file, 'r') as f:
                    val_ids = set(json.load(f))
            except Exception:
                val_ids = set()

            for t in all_tracks:
                if os.path.basename(t) in val_ids:
                    val_tracks.append(t)
                else:
                    train_tracks.append(t)
            
            # Check consistency: If val empty or no Scenario-B, recreate
            scenario_b_in_val = any("Scenario-B" in t for t in val_tracks)
            if not val_tracks or (not scenario_b_in_val and len(all_tracks) > 100):
                print("⚠️ Split file invalid or missing Scenario-B. Recreating...")
                val_tracks = []  # Reset to trigger new split logic

        # 2. Create new split if needed
        if not val_tracks:
            print("⚠️ Creating new split (Taking Val only from Scenario-B)...")
            
            # Filter Scenario-B tracks
            scenario_b_tracks = [t for t in all_tracks if "Scenario-B" in t]
            
            if not scenario_b_tracks:
                print("⚠️ Warning: No 'Scenario-B' folder found. Using random from all.")
                scenario_b_tracks = all_tracks
            
            # Val size = (1 - split_ratio) * total_scenario_b
            val_size = max(1, int(len(scenario_b_tracks) * (1 - split_ratio)))
            
            # Shuffle and take from beginning as val
            random.Random(self.seed).shuffle(scenario_b_tracks)
            val_tracks = scenario_b_tracks[:val_size]
            
            # Train = (All) - (Val)
            val_set = set(val_tracks)
            train_tracks = [t for t in all_tracks if t not in val_set]
            
            # Save track IDs (folder names)
            os.makedirs(os.path.dirname(self.val_split_file), exist_ok=True)
            with open(self.val_split_file, 'w') as f:
                json.dump([os.path.basename(t) for t in val_tracks], f, indent=2)

        return train_tracks, val_tracks

    def _index_samples(self, tracks: List[str]) -> None:
        """Index all samples from selected tracks."""
        for track_path in tqdm(tracks, desc=f"Indexing {self.mode}"):
            json_path = os.path.join(track_path, "annotations.json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    data = data[0]
                label = data.get('plate_text', data.get('license_plate', data.get('text', '')))
                if not label:
                    continue
                
                track_id = os.path.basename(track_path)
                
                lr_files = sorted(
                    glob.glob(os.path.join(track_path, "lr-*.png")) +
                    glob.glob(os.path.join(track_path, "lr-*.jpg"))
                )
                hr_files = sorted(
                    glob.glob(os.path.join(track_path, "hr-*.png")) +
                    glob.glob(os.path.join(track_path, "hr-*.jpg"))
                )
                
                # --- THAY ĐỔI Ở ĐÂY ---
                # Thay vì append 2 lần, ta chỉ append 1 lần chứa cả 2 loại đường dẫn
                self.samples.append({
                    'lr_paths': lr_files,  # Lưu đường dẫn LR
                    'hr_paths': hr_files,  # Lưu đường dẫn HR (có thể rỗng nếu không có)
                    'label': label,
                    'track_id': track_id
                })
                # ---------------------

            except Exception:
                pass
    def _index_test_samples(self, tracks: List[str]) -> None:
        """Index test samples without labels."""
        for track_path in tqdm(tracks, desc="Indexing test"):
            track_id = os.path.basename(track_path)
            
            # Load all LR images (sorted by frame number)
            lr_files = sorted(
                glob.glob(os.path.join(track_path, "lr-*.png")) +
                glob.glob(os.path.join(track_path, "lr-*.jpg"))
            )
            
            if lr_files:
                self.samples.append({
                    'lr_paths': lr_files,   # Đổi tên key từ 'paths' thành 'lr_paths'
                    'hr_paths': [],         # Thêm hr_paths rỗng để code không bị lỗi
                    'label': '',            # No label for test data
                    'track_id': track_id
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str, str]:
        item = self.samples[idx]
        
        # -----------------------------------------------------------
        # BƯỚC 1: QUYẾT ĐỊNH CHỌN ẢNH THẬT (LR) HAY GIẢ (HR)
        # -----------------------------------------------------------
        img_paths = item['lr_paths']
        is_synthetic = False

        # Logic Random 50/50: Chỉ áp dụng khi Train VÀ có ảnh HR
        if self.mode == 'train' and len(item['hr_paths']) > 0:
            if random.random() > 0.5: 
                img_paths = item['hr_paths']
                is_synthetic = True
        
        # -----------------------------------------------------------
        # BƯỚC 2: PADDING / TRUNCATE ĐỂ ĐẢM BẢO LUÔN ĐỦ 5 FRAMES
        # (Phải làm bước này SAU khi đã chọn được list ảnh cuối cùng)
        # -----------------------------------------------------------
        required_frames = 5
        if len(img_paths) < required_frames:
            # Nếu thiếu: Lặp lại frame cuối cùng cho đủ
            padding_count = required_frames - len(img_paths)
            if len(img_paths) > 0:
                img_paths = img_paths + [img_paths[-1]] * padding_count
            else:
                # Trường hợp cực đoan: track rỗng (không có ảnh nào)
                # Xử lý tạm: dùng đường dẫn 'dummy' để tạo ảnh đen ở bước sau
                img_paths = ["dummy_path.jpg"] * required_frames
        else:
            # Nếu thừa: Chỉ lấy 5 frame đầu
            img_paths = img_paths[:required_frames]

        # -----------------------------------------------------------
        # BƯỚC 3: LOAD ẢNH VÀ XỬ LÝ (DEGRADE + TRANSFORM)
        # -----------------------------------------------------------
        label = item['label']
        track_id = item['track_id']
        
        images_list = []
        for p in img_paths:
            # 3.1 Load ảnh
            image = cv2.imread(p, cv2.IMREAD_COLOR)
            
            # 3.2 Xử lý ảnh lỗi / không đọc được (hoặc dummy path)
            if image is None:
                image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 3.3 Làm xấu ảnh (Degradation) - Chỉ cho HR Synthetic
            if is_synthetic and self.degrade:
                image = self.degrade(image=image)['image']
            
            # 3.4 Augmentation / Normalize
            image = self.transform(image=image)['image']
            
            images_list.append(image)

        # Xếp chồng thành Tensor [5, 3, H, W]
        images_tensor = torch.stack(images_list, dim=0)
        
        # -----------------------------------------------------------
        # BƯỚC 4: XỬ LÝ NHÃN (LABEL)
        # -----------------------------------------------------------
        if self.is_test:
            target = [0]
            target_len = 1
        else:
            target = [self.char2idx[c] for c in label if c in self.char2idx]
            if len(target) == 0:
                target = [0]
            target_len = len(target)
            
        return images_tensor, torch.tensor(target, dtype=torch.long), target_len, label, track_id
    @staticmethod
    def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[str, ...], Tuple[str, ...]]:
        """Custom collate function for DataLoader."""
        images, targets, target_lengths, labels_text, track_ids = zip(*batch)
        images = torch.stack(images, 0)
        targets = torch.cat(targets)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        return images, targets, target_lengths, labels_text, track_ids
