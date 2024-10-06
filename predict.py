# predict.py

import torch
from torchvision import transforms
from PIL import Image
from model import BodyMeasurementModel
from constants import MEASUREMENT_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BodyMeasurementModel(num_measurements=len(MEASUREMENT_NAMES))
model.load_state_dict(
    torch.load(
        "body_measurement_model_combined_12measurements.pth", map_location=device
    )
)
model.to(device)
model.eval()


def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = transform(image).unsqueeze(0).to(device)
    return image


def predict_measurements(model, image, height):
    height_tensor = torch.tensor([[height]], dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = model(image, height_tensor)
    return predictions.squeeze().cpu().numpy()


def main():
    image_path = "my_image.jpg"
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image)
    height = 180.0  # Provide the height in cm

    measurements = predict_measurements(model, image_tensor, height)

    for name, measurement in zip(MEASUREMENT_NAMES, measurements):
        print(f"Predicted {name.replace('_', ' ').title()}: {measurement:.2f} cm")


if __name__ == "__main__":
    main()
