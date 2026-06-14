"""
config.py — Centralized configuration for the DeepGuard AI project.
All scripts import from here; change once, applies everywhere.

Optimized for 140K Real-and-Fake-Faces dataset (StyleGAN-generated faces).
"""

from pathlib import Path

# ─── Image / Model / Dataset ─────────────────────────────────────
MAX_TRAIN_SAMPLES = 50000
INPUT_SIZE = (224, 224, 3)
IMG_SIZE = INPUT_SIZE[:2]      # Compatible alias for legacy code
BATCH_SIZE = 32                # M1 Mac stable: reduced from 64 to prevent OOM + memory collisions
NUM_CLASSES = 1                # Binary: real vs fake

# ─── Training Phases ─────────────────────────────────────────────
PHASE_1_EPOCHS = 12
PHASE_1_LR = 3e-4

PHASE_2_EPOCHS = 20
PHASE_2_LR = 1e-5
UNFROZEN_LAYERS = 80  # Number of top layers to unfreeze in Phase 2

# Legacy aliases for training phases
EPOCHS_HEAD = PHASE_1_EPOCHS
EPOCHS_FINETUNE = PHASE_2_EPOCHS
LR_HEAD = PHASE_1_LR
LR_FINETUNE = PHASE_2_LR
FINETUNE_LAYERS = UNFROZEN_LAYERS

# ─── Optimization & Regularization ───────────────────────────────
OPTIMIZER = "Adam"  # Paired with Cosine Decay
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2
DROPOUT_HEAD = 0.5
DROPOUT_TAIL = 0.3

PHASE_1_PATIENCE = 5
PHASE_2_PATIENCE = 7

# ─── CBAM Attention / Architecture ───────────────────────────────
CBAM_REDUCTION_RATIO = 8
CBAM_RATIO = CBAM_REDUCTION_RATIO # Legacy alias

# ─── Paths ───────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR       = PROJECT_ROOT / "data" / "processed"
RAW_DIR        = PROJECT_ROOT / "data" / "raw"
REAL_RAW       = RAW_DIR / "real"
FAKE_RAW       = RAW_DIR / "fake"
REAL_PROCESSED = DATA_DIR / "real"
FAKE_PROCESSED = DATA_DIR / "fake"
MODELS_DIR     = PROJECT_ROOT / "models"

MODEL_PATH     = MODELS_DIR / "deepfake_detector.keras"   # primary (modern format)
MODEL_PATH_H5  = MODELS_DIR / "deepfake_detector_cbam.h5" # legacy fallback
HISTORY_PLOT   = MODELS_DIR / "training_history.png"

# ─── Chrome Extension ────────────────────────────────────────────
EXTENSION_DIR  = PROJECT_ROOT / "extension"
TFJS_MODEL_DIR = EXTENSION_DIR / "model"

# ─── Preprocessing ───────────────────────────────────────────────
FACE_PADDING   = 0.30     # 30% padding around face bounding box
FRAME_STEP     = 10       # Sample every Nth frame (≈3 fps at 30 fps)

# ─── Dataset ─────────────────────────────────────────────────────
DATASET_SLUG      = "xhlulu/140k-real-and-fake-faces"
DATASET_NAME      = "140K Real and Fake Faces"
# After download, the dataset provides train/valid/test splits:
#   train/ (real: 50K, fake: 50K)
#   valid/ (real: 10K, fake: 10K)
#   test/  (real: 10K, fake: 10K)
TRAIN_DIR         = DATA_DIR / "train"
VALID_DIR         = DATA_DIR / "valid"
TEST_DIR          = DATA_DIR / "test"
