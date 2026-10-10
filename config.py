import torch

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
VAL_RATIO = 0.1667

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "./data"
MODEL_SAVE_DIR = "./models"
RESULTS_SAVE_DIR = "./results"
