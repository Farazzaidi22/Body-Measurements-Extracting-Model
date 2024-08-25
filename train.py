import torch
from torchvision import transforms
from model import BodyMeasurementModel
from dataset import DeepFashionDataset
from torch.utils.data import DataLoader
from utils import train_one_epoch, validate


def main():
    # Set up data
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = DeepFashionDataset(
        root_dir="F:\\Upwork\\Pooya Kazerouni\\Body Measurements Extraction\\DeepFashion-MultiModal",
        transform=transform,
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Use num_workers to parallelize data loading, pin_memory speeds up the transfer to GPU
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    # Initialize model, optimizer, and loss function
    model = BodyMeasurementModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    # Training loop
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    # Save the model
    torch.save(model.state_dict(), "body_measurement_model.pth")


if __name__ == "__main__":
    main()
