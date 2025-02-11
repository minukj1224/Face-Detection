import os
import torch
import cv2
import config
import numpy as np
from model import ResNet18, ResNet34, ResNet50, ResNet101, DeepHourglassNet, DeepViT

device = config.DEVICE
model = config.MODEL().to(device)
model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
model.eval()

test_images = [os.path.join(config.DATASET_PATH, "test", f) for f in os.listdir(config.DATASET_PATH + "/test")]

for img_path in test_images:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, config.IMG_SIZE).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_resized).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor).cpu().numpy().reshape(-1, 2)

    for (x, y) in preds:
        cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

    cv2.imwrite(config.RANDOM_IMAGE_PATH + "/" + os.path.basename(img_path), img)