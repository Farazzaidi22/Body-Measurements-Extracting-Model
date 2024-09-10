import torch
from torchvision import transforms
from PIL import Image
from model import BodyMeasurementModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BodyMeasurementModel(num_measurements=2)
model.load_state_dict(torch.load("body_measurement_model.pth", map_location=device))
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
    image_path = "path/to/your/image.jpg"
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image)
    height = 180.34  # Provide the height in cm
    measurements = predict_measurements(model, image_tensor, height)
    print(f"Predicted shoulder length: {measurements[0]:.2f} cm")
    print(f"Predicted waist: {measurements[1]:.2f} cm")


if __name__ == "__main__":
    main()
