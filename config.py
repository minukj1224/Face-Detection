import torch
from model import ResNet18, ResNet34, ResNet50, ResNet101, DeepHourglassNet, DeepViT

DATASET_PATH = r"C:\Users\Minuk\Desktop\face_detection\dataset"
TRAIN_TXT = f"{DATASET_PATH}\\train\\train.txt"
VALIDATION_TXT = f"{DATASET_PATH}\\train\\test.txt"
TEST_IMAGE_PATH = r"C:\Users\Minuk\Desktop\face_detection\dataset\test"

IMG_SIZE = (96, 96)

MODEL = ResNet18
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

MODEL_SAVE_PATH = r"C:\Users\Minuk\Desktop\face_detection\ResNet18_models"
BEST_MODEL_PATH = f"{MODEL_SAVE_PATH}\\best_model.pth"
LAST_MODEL_PATH = f"{MODEL_SAVE_PATH}\\last_model.pth"

SCHEDULER_STEP_SIZE = 10
SCHEDULER_GAMMA = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"