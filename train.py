# train.py

import torch
from torchvision import transforms
from model import BodyMeasurementModel
from dataset import get_combined_dataset, collate_fn
from torch.utils.data import DataLoader
from utils import train_one_epoch, validate
from torch.cuda.amp import GradScaler

# Ensure you import MEASUREMENT_NAMES if you need to reference them
from constants import MEASUREMENT_NAMES


def main():
    # Set up data transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Get the combined dataset
    combined_dataset = get_combined_dataset(
        deepfashion_root="F:\\Upwork\\Pooya Kazerouni\\Body Measurements Extraction - DeepFashion 1\\DeepFashion-MultiModal",
        fashionpedia_root="F:\\Upwork\\Pooya Kazerouni\\Body Measurements Extraction - DeepFashion 1\\Fashionpedia",
        ochuman_root="F:\\Upwork\\Pooya Kazerouni\\Body Measurements Extraction - DeepFashion 1\\OCHuman",
        human_parts_root="F:\\Upwork\\Pooya Kazerouni\\Body Measurements Extraction - DeepFashion 1\\Priv_personpart",
        transform=transform,
    )

    print("Dataset loaded successfully.")

    # Split dataset into training and validation
    train_size = int(0.8 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,  # Use the custom collate function
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,  # Use the custom collate function
    )
    # Initialize the model
    model = BodyMeasurementModel(num_measurements=len(MEASUREMENT_NAMES))

    # Initialize the optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss(reduction="none")  # Use reduction='none' to apply mask

    # Use mixed precision for faster training
    scaler = GradScaler()

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Number of epochs
    num_epochs = 20

    # Gradient accumulation settings
    accumulation_steps = 2

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            accumulation_steps,
        )
        val_loss = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    # Save the model after training
    torch.save(model.state_dict(), "body_measurement_model_combined_12measurements.pth")


if __name__ == "__main__":
    main()
