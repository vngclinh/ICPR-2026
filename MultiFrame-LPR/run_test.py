import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Thêm đường dẫn để import được các module trong src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.restran import ResTranOCR
from src.utils.postprocess import decode_with_confidence
from src.utils.common import seed_everything

def run_inference():
    # 1. CẤU HÌNH (SETUP)
    # ---------------------------------------------------------
    TEST_DATA_PATH = "data/TKzFBtn7-test-blind/TKzFBtn7-test-blind"  # <-- Đường dẫn folder test của bạn
    CHECKPOINT_PATH = "results/restran_best.pth"   # <-- Đường dẫn model best
    OUTPUT_FILE = "results/submission_final.txt"   # File kết quả
    # ---------------------------------------------------------

    # Load config mặc định để lấy tham số model (img size, vocab, etc.)
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(config.SEED)

    print(f"🚀 START INFERENCE")
    print(f"• Data: {TEST_DATA_PATH}")
    print(f"• Model: {CHECKPOINT_PATH}")
    print(f"• Device: {device}")

    # 2. CHUẨN BỊ DỮ LIỆU (DATA LOADER)
    # ---------------------------------------------------------
    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ Error: Không tìm thấy folder data tại {TEST_DATA_PATH}")
        return

    test_dataset = MultiFrameDataset(
        root_dir=TEST_DATA_PATH,
        mode='val',         # Mode val để tắt augmentation
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX,
        is_test=True        # Quan trọng: Báo là test data (không có label)
    )

    if len(test_dataset) == 0:
        print("❌ Error: Không tìm thấy ảnh nào trong folder test.")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    print(f"✅ Loaded {len(test_dataset)} samples.")

    # 3. KHỞI TẠO MODEL & LOAD WEIGHT
    # ---------------------------------------------------------
    model = ResTranOCR(
        num_classes=config.NUM_CLASSES,
        transformer_heads=config.TRANSFORMER_HEADS,
        transformer_layers=config.TRANSFORMER_LAYERS,
        transformer_ff_dim=config.TRANSFORMER_FF_DIM,
        dropout=config.TRANSFORMER_DROPOUT,
        use_stn=config.USE_STN
    ).to(device)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Error: Không tìm thấy file weight tại {CHECKPOINT_PATH}")
        return

    # Load state dict
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        print("✅ Weights loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    # 4. CHẠY DỰ ĐOÁN (INFERENCE LOOP)
    # ---------------------------------------------------------
    model.eval()
    results = []
    
    # Mapping từ index số sang ký tự
    idx2char = config.IDX2CHAR

    print("🔮 Running prediction...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inferencing"):
            # Unpack batch (chú ý: MultiFrameDataset trả về 5 giá trị)
            images, _, _, _, track_ids = batch
            images = images.to(device)

            # Forward pass
            preds = model(images)  # Output: [Batch, Seq_Len, Classes]

            # Decode kết quả
            decoded_list = decode_with_confidence(preds, idx2char)

            # Lưu kết quả
            for i, (pred_text, conf) in enumerate(decoded_list):
                track_id = track_ids[i]
                results.append(f"{track_id},{pred_text};{conf:.4f}")

    # 5. LƯU FILE KẾT QUẢ
    # ---------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write("\n".join(results))
    
    print(f"\n🎉 DONE! Saved {len(results)} results to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_inference()