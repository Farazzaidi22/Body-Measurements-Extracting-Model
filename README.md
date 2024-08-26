````markdown
# Body Measurements Prediction Using DeepFashion Dataset

This project utilizes a Vision Transformer (ViT) model to predict body measurements such as shoulder length, waist size, and other clothing-related attributes from images. The model is trained on the DeepFashion Multi-Modal dataset and makes use of both image data and human annotations.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Information](#dataset-information)
3. [Installation and Setup](#installation-and-setup)
4. [Training the Model](#training-the-model)
5. [Predicting with the Model](#predicting-with-the-model)
6. [File Structure](#file-structure)
7. [Acknowledgments](#acknowledgments)

## Project Overview

The goal of this project is to predict various body measurements and clothing attributes from an image using a deep learning model based on the Vision Transformer (ViT) architecture. The project is built using `PyTorch` and Hugging Face's `transformers` library.

### What Does This Project Do?

- **Training**: The project trains a deep learning model to predict body measurements using the DeepFashion dataset.
- **Prediction**: Given an image and a height, the trained model can predict measurements such as shoulder length, waist size, sleeve length, and more.

## Dataset Information

The dataset used in this project is **DeepFashion-MultiModal**, which contains high-quality human images with various annotations including body measurements, clothing attributes, and more.

- **Dataset URL**: [DeepFashion Multi-Modal GitHub](https://github.com/yumingj/DeepFashion-MultiModal)
- **Download the Dataset**:
  - **Images**: ~5.4 GB, [Download here](https://drive.google.com)
  - **Annotations**: Shape, fabric, and pattern annotations available in TXT and JSON files.

### Folder Structure of the Dataset

- `images/`: Contains the images.
- `labels/`: Contains the annotation files such as `shape_anno_all.txt`, `fabric_ann.txt`, and `pattern_ann.txt`.

## Installation and Setup

### Python and CUDA Versions

- **Python Version**: 3.10 (Recommended)
- **CUDA Version**: 12.6.20 (Recommended for NVIDIA GPUs with CUDA support)
- **CUDA CNN Version**: Download cuDNN v8.9.7 for CUDA 12.x (Recommended for NVIDIA GPUs with CUDA support)
- **PyTorch Version**: `torch==2.0.1+cu117`
- **TorchVision Version**: `torchvision==0.15.2+cu117`

### Installing Dependencies

1. **Set up a Python virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
````

2. **Install PyTorch with CUDA Support**:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   ```

3. **Install other required dependencies**:

   ```bash
   pip install transformers==4.30.2 pillow==9.5.0 pandas==2.0.3 numpy==1.25.0 tqdm==4.65.0
   ```

4. **Ensure that CUDA is available**:
   After installing the dependencies, verify that PyTorch can detect your GPU:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should return True if GPU is available
   ```

### Requirements

A complete list of dependencies can be found in the `requirements.txt` file:

```plaintext
torch==2.0.1+cu117
torchvision==0.15.2+cu117
transformers==4.30.2
pillow==9.5.0
pandas==2.0.3
numpy==1.25.0
tqdm==4.65.0
```

### Download the Dataset

Download the dataset from the [DeepFashion-MultiModal GitHub repository](https://github.com/yumingj/DeepFashion-MultiModal).

1. **Images**: Download the images and extract them into a folder `images/`.
2. **Annotations**: Download the annotation files (shape, fabric, etc.) and place them inside the `labels/` directory.

The folder structure should look like this:

```plaintext
DeepFashion-MultiModal/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels/
│   ├── shape/
│   │   └── shape_anno_all.txt
│   ├── texture/
│   │   └── fabric_ann.txt
│   └── ...
└── captions.json
```

## Training the Model

To train the model, use the `train.py` script.

1. **Modify the paths**: Update the paths in `train.py` to point to the correct dataset location.
2. **Run the training script**:
   ```bash
   python train.py
   ```

The training script will:

- Load the dataset.
- Train the model using a Vision Transformer (ViT).
- Save the trained model to a file `body_measurement_model.pth`.

### Training Configuration

- **Batch Size**: 32
- **Epochs**: 5 (can be adjusted)
- **Optimizer**: AdamW
- **Loss Function**: Mean Squared Error (MSE)

## Predicting with the Model

After training, you can use the `predict.py` script to predict measurements from new images.

### Usage

1. **Modify `image_path` and `height` in `predict.py`** to point to your image and provide the height in cm.
2. **Run the prediction script**:
   ```bash
   python predict.py
   ```

This script will:

- Load the pre-trained model.
- Preprocess the input image.
- Use the model to predict body measurements (e.g., shoulder length, waist, etc.).
- Print the predicted values.

### Example Output

```plaintext
Predicted shoulder length: 42.50 cm
Predicted waist: 78.25 cm
Predicted sleeve length: 50.10 cm
...
```

## File Structure

Here’s an overview of the project files:

```
.
├── dataset.py                # Contains the dataset class for loading data
├── train.py                  # Script to train the model
├── predict.py                # Script to predict body measurements
├── model.py                  # Model definition using Vision Transformer (ViT)
├── utils.py                  # Training and validation helper functions
├── requirements.txt          # Python dependencies
└── README.md                 # This README file
```

## Acknowledgments

- **DeepFashion Dataset**: Special thanks to the authors of the [DeepFashion-MultiModal dataset](https://github.com/yumingj/DeepFashion-MultiModal) for providing the dataset used in this project.
- **Vision Transformer (ViT)**: This project leverages Hugging Face's `transformers` library for the pre-trained Vision Transformer model.

```
