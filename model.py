# model.py

import torch
import torch.nn as nn
from transformers import ViTModel


class BodyMeasurementModel(nn.Module):
    def __init__(self, num_measurements=12):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.fc1 = nn.Linear(self.vit.config.hidden_size + 1, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_measurements)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, image, height):
        vit_output = self.vit(image).last_hidden_state[:, 0]
        combined = torch.cat((vit_output, height), dim=1)
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        measurements = self.fc3(x)
        return measurements
