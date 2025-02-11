import cv2
import os
import random
import torch
import numpy as np
import config
from model import ResNet18, ResNet34, ResNet50, ResNet101, DeepHourglassNet, DeepViT

def visualize_and_save_random_image(model, epoch):
    img_list = [f for f in os.listdir(config.TEST_IMAGE_PATH) if f.endswith(('.jpg', '.png'))]
    if not img_list:
        print("🔴 랜덤 이미지 저장 실패: 테스트 폴더에 이미지가 없습니다.")
        return
    
    img_name = random.choice(img_list)
    img_path = os.path.join(config.TEST_IMAGE_PATH, img_name)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, config.IMG_SIZE)
    orig_img = img.copy()
    img_resized = cv2.resize(img, config.IMG_SIZE).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_resized).unsqueeze(0).unsqueeze(0).to(config.DEVICE)

    model.eval()
    with torch.no_grad():
        preds = model(img_tensor).cpu().numpy().reshape(-1, 2)

    for (x, y) in preds:
        cv2.circle(orig_img, (int(x), int(y)), 2, (0, 0, 255), -1)

    save_path = os.path.join(config.MODEL_SAVE_PATH, f"epoch_{epoch+1}_result.png")
    cv2.imwrite(save_path, orig_img)
    print(f"✅ 랜덤 이미지 예측 결과 저장 완료: {save_path}")