import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, heights, true_measurements, _ in tqdm(dataloader):
        # Ensure all tensors are moved to the GPU
        images, heights, true_measurements = images.to(device, non_blocking=True), heights.to(device, non_blocking=True), true_measurements.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        predicted_measurements = model(images, heights)
        loss = criterion(predicted_measurements, true_measurements)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, heights, true_measurements, _ in tqdm(dataloader):
            # Ensure all tensors are moved to the GPU
            images, heights, true_measurements = images.to(device, non_blocking=True), heights.to(device, non_blocking=True), true_measurements.to(device, non_blocking=True)
            
            predicted_measurements = model(images, heights)
            loss = criterion(predicted_measurements, true_measurements)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)
