import torch
from torchvision import transforms
from PIL import Image
from model import BodyMeasurementModel


def predict_measurements(model, image_path, height):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Image preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Prepare the height tensor
    height_tensor = torch.tensor([[height]], dtype=torch.float32).to(device)

    # Make predictions
    with torch.no_grad():
        predictions = model(image, height_tensor)

    return predictions.squeeze().cpu().numpy()


def main():
    # Load the pre-trained model
    model = BodyMeasurementModel()
    model.load_state_dict(torch.load("body_measurement_model.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Example usage: provide your own image and height
    image_path = "my_image_2.jpg"  # Path to your image
    height = 180.33999999999997  # Provide the height in cm

    # Get predictions for shoulder length and waist
    measurements = predict_measurements(model, image_path, height)
    print(f"Predicted shoulder length: {measurements[0]:.2f} cm")
    print(f"Predicted waist: {measurements[1]:.2f} cm")


if __name__ == "__main__":
    main()
