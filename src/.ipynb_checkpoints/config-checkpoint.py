"""Configuration settings for the spelling correction model."""

import torch

# General settings
SEED = 42
MAX_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
HIDDEN_DIM = 256
EMBEDDING_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
TEACHER_FORCING_RATIO = 0.5

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 4
CLIP_GRAD = 1.0
VAL_INTERVAL = 1000  # Validate every N batches
SAVE_INTERVAL = 5000  # Save checkpoint every N batches

# Paths
TRAIN_DATA_PATH = "data/train.json"
VAL_DATA_PATH = "data/test.json"
TOKENIZER_VOCAB_PATH = "tokenizer_vocab.json"
CHECKPOINT_DIR = "checkpoints"