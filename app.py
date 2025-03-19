import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import streamlit as st

# Define paths
TRAIN_PATH = "train_data"
MODEL_PATH = "siamese_effnet_lite0.pth"

# Image preprocessing function
def preprocess_image(image):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    final_pil = Image.fromarray(edges)
    return final_pil

# Data augmentation function
def augment_image(image):
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])
    return [augmentations(image) for _ in range(5)]

# Transformation pipeline for grayscale images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Define Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        base_model = models.efficientnet_b0(pretrained=True)
        self.cnn = base_model.features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.cnn[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

    def forward_one(self, x):
        x = self.cnn(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

# Load model
@st.cache_resource
def load_model():
    model = SiameseNetwork()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

def compute_similarity(img1, img2):
    img1, img2 = transform(img1).unsqueeze(0), transform(img2).unsqueeze(0)
    with torch.no_grad():
        output1, output2 = model(img1, img2)
        distance = torch.nn.functional.pairwise_distance(output1, output2).item()
    return distance

# Add new tool with augmentation
def add_new_tool(image, tool_name):
    tool_dir = os.path.join(TRAIN_PATH, tool_name)
    os.makedirs(tool_dir, exist_ok=True)
    augmented_images = augment_image(image)
    image.save(os.path.join(tool_dir, "original.jpg"))
    for i, aug_image in enumerate(augmented_images):
        aug_image.save(os.path.join(tool_dir, f"aug_{i}.jpg"))
    st.success(f"Added new tool: {tool_name}")

# Scan tool dynamically
def scan_tool(image):
    image = preprocess_image(image)
    best_match, best_distance = None, float('inf')
    for tool_name in os.listdir(TRAIN_PATH):
        tool_dir = os.path.join(TRAIN_PATH, tool_name)
        for filename in os.listdir(tool_dir):
            tool_img = Image.open(os.path.join(tool_dir, filename)).convert('L')
            distance = compute_similarity(image, tool_img)
            if distance < best_distance:
                best_distance, best_match = distance, tool_name
    return best_match if best_distance < 1.0 else "Unknown Tool"

# Streamlit UI
st.title("Siamese Network Tool Recognition")

option = st.radio("Choose an action", ["Scan Tool", "Add New Tool"])

if option == "Scan Tool":
    uploaded_file = st.file_uploader("Upload tool image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Tool", use_column_width=True)
        if st.button("Identify Tool"):
            result = scan_tool(image)
            st.success(f"Identified as: {result}")

elif option == "Add New Tool":
    uploaded_file = st.file_uploader("Upload new tool image", type=["jpg", "png", "jpeg"])
    tool_name = st.text_input("Enter tool name")
    if uploaded_file and tool_name:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Tool", use_column_width=True)
        if st.button("Add Tool"):
            image = preprocess_image(image)
            add_new_tool(image, tool_name)
