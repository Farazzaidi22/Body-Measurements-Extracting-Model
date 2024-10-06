# dataset.py

import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
import xml.etree.ElementTree as ET
from constants import MEASUREMENT_NAMES


class DeepFashionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.captions_file = os.path.join(root_dir, "captions.json")
        self.shape_file = os.path.join(root_dir, "labels/shape/shape_anno_all.txt")

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
                # Other attributes...

                # Map the annotations to features or measurements
                height = self.map_to_height(sleeve_length, lower_clothing_length)
                measurement_data = {
                    "height": height,
                    "shoulder_length": self.map_to_shoulder_length(sleeve_length),
                    "waist": self.map_to_waist(lower_clothing_length),
                    # Initialize other measurements as -1 (unknown)
                }

                # Initialize other measurements to -1
                for name in MEASUREMENT_NAMES:
                    if name not in measurement_data:
                        measurement_data[name] = -1  # Unknown measurement

                measurements[img_name] = measurement_data
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
        measurement_data = self.measurements.get(img_name, {})
        height = measurement_data.get("height", 170.0)  # Default height

        # Initialize measurements and mask
        measurements = []
        mask = []
        for name in MEASUREMENT_NAMES:
            value = measurement_data.get(name, -1)
            measurements.append(value)
            mask.append(0.0 if value == -1 else 1.0)
        measurements = torch.tensor(measurements, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return (
            image,
            torch.tensor([height], dtype=torch.float32),
            measurements,
            mask,
            caption,
        )


class FashionpediaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.annotations_file = os.path.join(
            root_dir, "instances_attributes_train2020.json"
        )

        with open(self.annotations_file, "r") as f:
            self.data = json.load(f)

        self.image_info = self.data["images"]
        self.annotations = self.data["annotations"]

        # Create a mapping from image_id to annotations
        self.image_to_annotations = {}
        for ann in self.annotations:
            image_id = ann["image_id"]
            if image_id not in self.image_to_annotations:
                self.image_to_annotations[image_id] = []
            self.image_to_annotations[image_id].append(ann)

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        img_info = self.image_info[idx]
        img_path = os.path.join(self.root_dir, "train", img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        # Get annotations for this image
        annotations = self.image_to_annotations.get(img_info["id"], [])

        # Extract measurements
        # Assuming height is provided in the image metadata or annotations
        height = img_info.get(
            "height_cm", 170.0
        )  # Replace 'height_cm' with the correct key if available

        measurement_data = {
            name: -1 for name in MEASUREMENT_NAMES
        }  # Initialize with -1

        for ann in annotations:
            if "attributes" in ann:
                attributes = ann["attributes"]
                # Map attributes to measurements
                # Replace with actual attribute indices or names
                measurement_data["shoulder_length"] = attributes.get(
                    "shoulder_length", -1
                )
                measurement_data["waist"] = attributes.get("waist_circumference", -1)
                # Add mappings for other measurements if available

        # Initialize measurements and mask
        measurements = []
        mask = []
        for name in MEASUREMENT_NAMES:
            value = measurement_data.get(name, -1)
            measurements.append(value)
            mask.append(0.0 if value == -1 else 1.0)
        measurements = torch.tensor(measurements, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return (
            image,
            torch.tensor([height], dtype=torch.float32),
            measurements,
            mask,
            "",
        )


class OCHumanApiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        import copy
        import os

        self.root_dir = root_dir
        self.transform = transform

        # List of possible annotation filenames
        possible_annotation_files = [
            "ochuman_coco_format_test_range_0.00_1.00.json",
            "ochuman_coco_format_val_range_0.00_1.00.json",
            # "ochuman_coco_format_train_range_0.00_1.00.json",
            "ochuman.json",
        ]

        # Find the first existing annotation file
        annotation_file = None
        for fname in possible_annotation_files:
            fpath = os.path.join(root_dir, fname)
            if os.path.isfile(fpath):
                annotation_file = fpath
                break

        if annotation_file is None:
            raise FileNotFoundError(
                f"No annotation file found in {root_dir}. Expected one of: {', '.join(possible_annotation_files)}"
            )

        print(f"Using annotation file: {annotation_file}")

        with open(annotation_file, "r") as f:
            self.data = json.load(f)

        # Check the structure of self.data
        if (
            isinstance(self.data, dict)
            and "images" in self.data
            and "annotations" in self.data
        ):
            print("Data is in COCO format")
            self.image_info = self.data["images"]
            self.annotations = self.data["annotations"]
        elif isinstance(self.data, list):
            print("Data is a list of image dictionaries")
            self.image_info = self.data
            self.annotations = []

            for idx, img in enumerate(self.image_info):
                img_copy = copy.deepcopy(img)
                image_id = img_copy.get("id", idx)
                img_copy["id"] = image_id

                annotations = img_copy.get("annotations") or img_copy.get(
                    "human_annotations", []
                )
                if isinstance(annotations, dict):
                    annotations = [annotations]
                for ann in annotations:
                    ann["image_id"] = image_id
                    self.annotations.append(ann)

                self.image_info[idx] = img_copy
        else:
            raise ValueError(
                f"Unsupported data format in the annotation file: {annotation_file}. Data structure: {type(self.data)}"
            )

        # Create a mapping from image_id to annotations
        self.image_to_annotations = {}
        for ann in self.annotations:
            image_id = ann["image_id"]
            if image_id not in self.image_to_annotations:
                self.image_to_annotations[image_id] = []
            self.image_to_annotations[image_id].append(ann)

        print(
            f"Loaded {len(self.image_info)} images and {len(self.annotations)} annotations"
        )

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        img_info = self.image_info[idx]
        img_filename = img_info.get("file_name") or img_info.get("filepath")
        if img_filename is None:
            raise KeyError("Image filename not found in image info.")
        img_path = os.path.join(self.root_dir, "images", img_filename)
        image = Image.open(img_path).convert("RGB")

        # Get annotations for this image
        image_id = img_info.get("id", idx)
        annotations = self.image_to_annotations.get(image_id, [])

        # Initialize measurements
        height = img_info.get("height_cm", 170.0)  # Default height if not provided
        measurement_data = {
            name: -1 for name in MEASUREMENT_NAMES
        }  # Initialize with -1

        for ann in annotations:
            # Extract measurements from annotations if available
            if "keypoints" in ann:
                keypoints = torch.tensor(ann["keypoints"]).reshape(-1, 3)

                # Example calculation of height using keypoints
                nose = keypoints[0]
                left_ankle = keypoints[15]
                right_ankle = keypoints[16]
                height = max(
                    abs(nose[1] - left_ankle[1]), abs(nose[1] - right_ankle[1])
                )

                # Calculate other measurements if possible
                # For now, we'll break after calculating height
                break

        # Initialize measurements and mask
        measurements = []
        mask = []
        for name in MEASUREMENT_NAMES:
            value = measurement_data.get(name, -1)
            measurements.append(value)
            mask.append(0.0 if value == -1 else 1.0)
        measurements = torch.tensor(measurements, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return (
            image,
            torch.tensor([height], dtype=torch.float32),
            measurements,
            mask,
            "",
        )


class HumanPartsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.annotations_dir = os.path.join(root_dir, "Annotations")
        self.image_dir = os.path.join(root_dir, "Images")
        self.annotation_files = [
            os.path.join(self.annotations_dir, f)
            for f in os.listdir(self.annotations_dir)
            if f.endswith(".xml")
        ]
        self.image_extensions = [".jpg", ".jpeg", ".png"]

    def __len__(self):
        return len(self.annotation_files)

    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        file_name = root.find("filename").text
        file_name = file_name.lower()

        height = None
        measurement_data = {name: -1 for name in MEASUREMENT_NAMES}

        for obj in root.findall("object"):
            label = obj.find("name").text
            bndbox = obj.find("bndbox")
            ymin = int(bndbox.find("ymin").text)
            ymax = int(bndbox.find("ymax").text)

            if label in measurement_data:
                measurement_data[label] = ymax - ymin

            if label == "person" and height is None:
                height = ymax - ymin

        if height is None:
            height = 170.0  # Default height

        return file_name, height, measurement_data

    def find_image(self, file_name):
        for ext in self.image_extensions:
            img_path = os.path.join(self.image_dir, file_name + ext)
            if os.path.exists(img_path):
                return img_path
        return None

    def __getitem__(self, idx):
        annotation_path = self.annotation_files[idx]
        file_name, height, measurement_data = self.parse_annotation(annotation_path)

        img_path = self.find_image(file_name)
        if img_path is None:
            # Log the missing file and skip this example
            print(f"Warning: Image {file_name} not found. Skipping.")
            return None  # This will allow us to skip the entry

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Log error if there's an issue opening the image
            print(f"Error loading image {file_name}: {e}")
            return None

        measurements = []
        mask = []
        for name in MEASUREMENT_NAMES:
            value = measurement_data.get(name, -1)
            measurements.append(value)
            mask.append(0.0 if value == -1 else 1.0)

        measurements = torch.tensor(measurements, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        height_tensor = torch.tensor([height], dtype=torch.float32)

        return image, height_tensor, measurements, mask, ""


def collate_fn(batch):
    # Filter out None values from the batch (skipped items)
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None, None, None, None
    return torch.utils.data.dataloader.default_collate(batch)


def get_combined_dataset(
    deepfashion_root, fashionpedia_root, ochuman_root, human_parts_root, transform=None
):
    deepfashion_dataset = DeepFashionDataset(deepfashion_root, transform)
    fashionpedia_dataset = FashionpediaDataset(fashionpedia_root, transform)
    ochuman_dataset = OCHumanApiDataset(ochuman_root, transform)
    human_parts_dataset = HumanPartsDataset(human_parts_root, transform)
    return ConcatDataset(
        [
            deepfashion_dataset,
            fashionpedia_dataset,
            ochuman_dataset,
            human_parts_dataset,
        ]
    )
