import torch
from torchvision import transforms
from PIL import Image
from model import BodyMeasurementModel

def predict_measurements(model, image_path, height):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    height = torch.tensor([[height]], dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(image, height)

    return predictions.squeeze().cpu().numpy()

def main():
    model = BodyMeasurementModel()
    model.load_state_dict(torch.load('body_measurement_model.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    image_path = 'path/to/test/image.jpg'
    height = 175  # in cm
    measurements = predict_measurements(model, image_path, height)
    print(f"Predicted shoulder length: {measurements[0]:.2f} cm")
    print(f"Predicted waist: {measurements[1]:.2f} cm")

if __name__ == "__main__":
    main()