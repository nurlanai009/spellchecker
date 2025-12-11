"""Configuration settings for the improved spelling correction model."""

import torch

# General settings
SEED = 42
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
HIDDEN_DIM = 256  # Must be divisible by N_HEADS
EMBEDDING_DIM = 256
NUM_LAYERS = 3
DROPOUT = 0.3
N_HEADS = 8  # Number of attention heads
TEACHER_FORCING_RATIO = 0.3

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 15
CLIP_GRAD = 1.0
VAL_INTERVAL = 4000  # Validate every N batches
SAVE_INTERVAL = 10000  # Save checkpoint every N batches


# Paths
TOKENIZER_VOCAB_PATH = "vocab.json"
CHECKPOINT_DIR = "models"