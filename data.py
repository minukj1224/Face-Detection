import torch
import cv2
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import config

class FaceDataset(Dataset):
    def __init__(self, txt_file):
        self.data = []
        with open(txt_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                img_path = os.path.join(config.DATASET_PATH, parts[0])
                landmarks = np.array(list(map(float, parts[5:])), dtype=np.float32).reshape(-1, 2)
                self.data.append((img_path, landmarks))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, landmarks = self.data[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, config.IMG_SIZE)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return torch.tensor(img), torch.tensor(landmarks).flatten()

def get_dataloaders():
    train_dataset = FaceDataset(config.TRAIN_TXT)
    test_dataset = FaceDataset(config.VALIDATION_TXT)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    return train_loader, test_loader