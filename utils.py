# utils.py

import torch
from tqdm import tqdm
from torch.cuda.amp import autocast


def train_one_epoch(
    model, dataloader, optimizer, criterion, device, scaler, accumulation_steps
):
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    optimizer.zero_grad()
    for i, (images, heights, measurements, masks, _) in enumerate(progress_bar):
        images = images.to(device, non_blocking=True)
        heights = heights.to(device, non_blocking=True)
        measurements = measurements.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast():
            outputs = model(images, heights)
            loss = (criterion(outputs, measurements) * masks).sum() / masks.sum()
            loss = loss / accumulation_steps  # Adjust for gradient accumulation

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()

        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Validating", leave=False)

    with torch.no_grad():
        for images, heights, measurements, masks, _ in progress_bar:
            images = images.to(device, non_blocking=True)
            heights = heights.to(device, non_blocking=True)
            measurements = measurements.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images, heights)
            loss = (criterion(outputs, measurements) * masks).sum() / masks.sum()

            total_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)
