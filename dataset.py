import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class DeepFashionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.captions_file = os.path.join(
            root_dir, "captions.json"
        )  # Example path for captions
        self.shape_file = os.path.join(
            root_dir, "labels/shape/shape_anno_all.txt"
        )  # Shape annotations

        # Load captions/annotations
        with open(self.captions_file, "r") as f:
            self.annotations = json.load(f)

        # Load shape annotations for measurements
        self.measurements = self.load_shape_annotations(self.shape_file)
        self.image_list = list(self.annotations.keys())

    def load_shape_annotations(self, shape_file):
        measurements = {}
        with open(shape_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                img_name = parts[0]

                # Extract the relevant shape annotations
                sleeve_length = int(parts[1])
                lower_clothing_length = int(parts[2])
                socks = int(parts[3])
                hat = int(parts[4])
                glasses = int(parts[5])
                neckwear = int(parts[6])
                wrist_wearing = int(parts[7])
                ring = int(parts[8])
                waist_accessories = int(parts[9])
                neckline = int(parts[10])
                outer_clothing_cardigan = int(parts[11])
                upper_clothing_covering_navel = int(parts[12])

                # Map the annotations to features or measurements
                height = self.map_to_height(sleeve_length, lower_clothing_length)
                shoulder_length = self.map_to_shoulder_length(sleeve_length)
                waist = self.map_to_waist(lower_clothing_length)

                measurements[img_name] = {
                    "height": height,
                    "shoulder_length": shoulder_length,
                    "waist": waist,
                    "sleeve_length": sleeve_length,
                    "lower_clothing_length": lower_clothing_length,
                    "socks": socks,
                    "hat": hat,
                    "glasses": glasses,
                    "neckwear": neckwear,
                    "wrist_wearing": wrist_wearing,
                    "ring": ring,
                    "waist_accessories": waist_accessories,
                    "neckline": neckline,
                    "outer_clothing_cardigan": outer_clothing_cardigan,
                    "upper_clothing_covering_navel": upper_clothing_covering_navel,
                }
        return measurements

    def map_to_height(self, sleeve_length, lower_clothing_length):
        # Example mapping based on sleeve and clothing length
        if sleeve_length == 0:  # Sleeveless
            return 150.0
        elif sleeve_length == 3:  # Long sleeve
            return 175.0
        elif lower_clothing_length == 3:  # Long lower clothing
            return 180.0
        else:
            return 160.0

    def map_to_shoulder_length(self, sleeve_length):
        # Example mapping of sleeve length to shoulder length
        if sleeve_length == 0:  # Sleeveless
            return 40.0
        elif sleeve_length == 3:  # Long sleeve
            return 50.0
        else:
            return 45.0

    def map_to_waist(self, lower_clothing_length):
        # Example mapping of clothing length to waist measurement
        if lower_clothing_length == 0:  # Three-point
            return 60.0
        elif lower_clothing_length == 3:  # Long clothing
            return 75.0
        else:
            return 70.0

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.root_dir, "images", img_name)
        image = Image.open(img_path).convert("RGB")

        caption = self.annotations.get(img_name, "")

        # Retrieve actual measurements from the shape annotations file
        measurement_data = self.measurements.get(img_name, None)
        if measurement_data:
            height = measurement_data["height"]
            shoulder_length = measurement_data["shoulder_length"]
            waist = measurement_data["waist"]
        else:
            raise ValueError(f"Measurements not found for image {img_name}")

        measurements = torch.tensor([shoulder_length, waist], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([height], dtype=torch.float32), measurements, caption
