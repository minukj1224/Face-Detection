import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import config
from model import ResNet18, ResNet34, ResNet50, ResNet101, DeepHourglassNet, DeepViT
from data import get_dataloaders
import os
from util import visualize_and_save_random_image

device = config.DEVICE
model = config.MODEL().to(device)
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.SCHEDULER_STEP_SIZE, gamma=config.SCHEDULER_GAMMA)
criterion = nn.MSELoss()

train_loader, test_loader = get_dataloaders()

os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)

best_loss = float("inf")

for epoch in range(config.EPOCHS):
    model.train()
    running_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.EPOCHS}]", leave=False)
    
    for images, landmarks in loop:
        images, landmarks = images.to(device), landmarks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, landmarks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), config.LAST_MODEL_PATH)
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), config.BEST_MODEL_PATH)

    visualize_and_save_random_image(model, epoch)

    scheduler.step()