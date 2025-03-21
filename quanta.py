import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import streamlit as st
import time
from collections import Counter
import pandas as pd

# Define paths
REFERENCE_PATH = "reference_data"
CLASSES = []

# Load available classes on startup
def load_available_classes():
    """Load available classes from disk and update CLASSES global variable"""
    global CLASSES
    CLASSES = []
    if os.path.exists(REFERENCE_PATH):
        for item in os.listdir(REFERENCE_PATH):
            item_path = os.path.join(REFERENCE_PATH, item)
            if os.path.isdir(item_path) and any(file.endswith('.pt') for file in os.listdir(item_path)):
                CLASSES.append(item)
    return CLASSES

# Multiple image preprocessing functions for ensemble approach
def preprocess_clahe_edge(image):
    """Preprocessing with CLAHE enhancement and edge detection"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 50, 150)
    return Image.fromarray(edges)

def preprocess_adaptive_threshold(image):
    """Preprocessing with adaptive thresholding"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return Image.fromarray(thresh)

def preprocess_sobel_filter(image):
    """Preprocessing with Sobel filter for edge detection"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return Image.fromarray(magnitude)

def preprocess_morphological(image):
    """Preprocessing with morphological operations"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    return Image.fromarray(opening)

def preprocess_combined_edges(image):
    """Preprocessing combining multiple edge detection methods"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges_canny = cv2.Canny(enhanced, 30, 150)
    laplacian = cv2.Laplacian(enhanced, cv2.CV_8U)
    combined = cv2.bitwise_or(edges_canny, laplacian)
    return Image.fromarray(combined)

# List all preprocessing functions for ensemble
PREPROCESSING_FUNCTIONS = [
    preprocess_clahe_edge,
    preprocess_adaptive_threshold,
    preprocess_sobel_filter,
    preprocess_morphological,
    preprocess_combined_edges
]

# Define transformation for RGB images (EfficientNet expects RGB)
transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define transformation for grayscale images (convert to RGB first)
transform_gray = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Feature extractor class using pre-trained EfficientNet
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = models.efficientnet_b0(weights='DEFAULT')
        self.features = nn.Sequential(*list(self.model.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

# Load model with caching
@st.cache_resource
def load_feature_extractor():
    model = FeatureExtractor()
    model.eval()
    return model

# Function to extract features from an image
def extract_features(model, image):
    """Extract feature vector from image using pre-trained model"""
    if isinstance(image, np.ndarray) and len(image.shape) == 2:
        image = Image.fromarray(image)
        tensor = transform_gray(image).unsqueeze(0)
    elif isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 1:
        image = Image.fromarray(image.squeeze())
        tensor = transform_gray(image).unsqueeze(0)
    elif isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 3:
        image = Image.fromarray(image)
        tensor = transform_rgb(image).unsqueeze(0)
    elif isinstance(image, Image.Image):
        if image.mode == 'L':
            tensor = transform_gray(image).unsqueeze(0)
        else:
            tensor = transform_rgb(image).unsqueeze(0)
    else:
        raise ValueError("Unsupported image format")

    with torch.no_grad():
        features = model(tensor)

    return features

# Function to compute similarity between feature vectors
def compute_similarity(features1, features2):
    """Compute cosine similarity between feature vectors"""
    similarity = torch.nn.functional.cosine_similarity(features1, features2).item()
    return similarity

# Function to add new reference tool
def add_new_tool(model, image, tool_name):
    """Add a new tool reference with multiple preprocessing techniques"""
    tool_dir = os.path.join(REFERENCE_PATH, tool_name)
    os.makedirs(tool_dir, exist_ok=True)
    base_filename = str(int(time.time()))
    feature_count = 0
    for i, preprocess_func in enumerate(PREPROCESSING_FUNCTIONS):
        try:
            processed_image = preprocess_func(image)
            features = extract_features(model, processed_image)
            torch.save(features, os.path.join(tool_dir, f"{base_filename}_aug{i}.pt"))
            processed_image.save(os.path.join(tool_dir, f"{base_filename}_aug{i}.jpg"))
            feature_count += 1
        except Exception as e:
            st.error(f"Error processing with augmentation {i}: {e}")
            continue
    if tool_name not in CLASSES:
        CLASSES.append(tool_name)
    return feature_count

# Function to get updated class list
def get_available_classes():
    """Get list of available classes by scanning the reference directory"""
    global CLASSES
    if not os.path.exists(REFERENCE_PATH):
        CLASSES = []
        return []
    classes = []
    for item in os.listdir(REFERENCE_PATH):
        item_path = os.path.join(REFERENCE_PATH, item)
        if os.path.isdir(item_path) and any(file.endswith('.pt') for file in os.listdir(item_path)):
            classes.append(item)
    CLASSES = classes
    return classes

# Function to scan a frame with multiple augmentations and ensemble voting
def scan_frame_ensemble(model, frame):
    """Scan a frame with multiple preprocessing techniques and ensemble voting"""
    available_classes = get_available_classes()
    if not available_classes:
        return "No reference data available", [], 0, []
    augmentation_results = []
    processed_images = []
    for preprocess_func in PREPROCESSING_FUNCTIONS:
        try:
            processed_image = preprocess_func(frame)
            processed_images.append(processed_image)
            features = extract_features(model, processed_image)
            similarities = []
            for tool_name in available_classes:
                tool_dir = os.path.join(REFERENCE_PATH, tool_name)
                if os.path.exists(tool_dir):
                    tool_similarities = []
                    for filename in os.listdir(tool_dir):
                        if filename.endswith('.pt'):
                            try:
                                feature_path = os.path.join(tool_dir, filename)
                                ref_features = torch.load(feature_path)
                                similarity = compute_similarity(features, ref_features)
                                tool_similarities.append((tool_name, similarity, filename))
                            except Exception as e:
                                continue
                    if tool_similarities:
                        tool_similarities.sort(key=lambda x: x[1], reverse=True)
                        similarities.append(tool_similarities[0])
            if similarities:
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_match = similarities[0]
                result = top_match[0]
                confidence = top_match[1]
                augmentation_results.append((result, confidence, similarities[:3]))
        except Exception as e:
            continue
    if not augmentation_results:
        return "Unknown Tool", [], 0, processed_images
    vote_results = {}
    for result, confidence, _ in augmentation_results:
        if result in vote_results:
            vote_results[result] += confidence
        else:
            vote_results[result] = confidence
    if vote_results:
        ensemble_result = max(vote_results.items(), key=lambda x: x[1])
        final_result = ensemble_result[0]
        matching_confidences = [conf for res, conf, _ in augmentation_results if res == final_result]
        avg_confidence = sum(matching_confidences) / len(matching_confidences) if matching_confidences else 0
        top_similarities = []
        for _, _, similarities in augmentation_results:
            top_similarities.extend([s for s in similarities if s[0] == final_result])
        top_similarities.sort(key=lambda x: x[1], reverse=True)
        top_5 = top_similarities[:5] if top_similarities else []
        return final_result, top_5, avg_confidence, processed_images
    else:
        return "Unknown Tool", [], 0, processed_images

# Main Streamlit UI with enhanced live video streaming
def main():
    st.title("Tool Recognition System")
    load_available_classes()
    model = load_feature_extractor()
    if not os.path.exists(REFERENCE_PATH):
        os.makedirs(REFERENCE_PATH)
    available_classes = get_available_classes()
    tab1, tab2, tab3 = st.tabs(["Live Stream", "Add Reference Tool", "Settings"])
    with tab1:
        if not available_classes:
            st.warning("No reference data available. Please add reference tools first.")
        run_livestream = st.checkbox("Start Live Stream")
        camera_id = 0
        stream_placeholder = st.empty()
        confidence_threshold = 0.35
        if run_livestream:
            with st.spinner("Starting live stream..."):
                cap = cv2.VideoCapture(int(camera_id))
                if not cap.isOpened():
                    st.error("Unable to open camera. Please check camera ID.")
                    run_livestream = False
                else:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    frame_count = 1
                    process_interval = 1
                    last_result = "Waiting for detection..."
                    result_window = []
                    window_size = 1
                    while run_livestream:
                        if frame_count % 100 == 0:
                            available_classes = get_available_classes()
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Error reading from camera.")
                            break
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        annotated_frame = frame_rgb.copy()
                        if frame_count % process_interval == 0:
                            pil_frame = Image.fromarray(frame_rgb)
                            result, top_5, confidence, _ = scan_frame_ensemble(model, pil_frame)
                            effective_result = result if confidence > confidence_threshold else "Unknown Tool"
                            result_window.append((effective_result, confidence))
                            if len(result_window) > window_size:
                                result_window.pop(0)
                            if result_window:
                                valid_results = [(r, c) for r, c in result_window if r != "Unknown Tool" and c > confidence_threshold]
                                if valid_results:
                                    result_counts = Counter([r for r, _ in valid_results])
                                    most_common = result_counts.most_common(1)[0]
                                    last_result = most_common[0]
                                else:
                                    last_result = "Unknown Tool"
                            overlay = annotated_frame.copy()
                            cv2.rectangle(overlay, (0, 0), (annotated_frame.shape[1], 60), (0, 0, 0), -1)
                            cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                            text_color = (0, 255, 0) if last_result != "Unknown Tool" else (255, 0, 0)
                            text = last_result
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                            text_x = (annotated_frame.shape[1] - text_size[0]) // 2
                            cv2.putText(annotated_frame, text, (text_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2)
                        with stream_placeholder.container():
                            st.image(annotated_frame, use_column_width=True)
                        frame_count += 1
                        time.sleep(0.03)
                        if not st.session_state.get("run_livestream", run_livestream):
                            break
                    cap.release()
    with tab2:
        st.header("Add Reference Tool")
        uploaded_file = st.file_uploader("Upload reference tool image", type=["jpg", "png", "jpeg"], key="new_tool")
        tool_name = st.text_input("Enter tool name (e.g., hammer, screwdriver)").lower()
        if uploaded_file and tool_name:
            image = Image.open(uploaded_file).convert('RGB')
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            if st.button("Add Reference Tool"):
                with st.spinner("Processing and extracting features with multiple augmentations..."):
                    num_features = add_new_tool(model, image, tool_name)
                    available_classes = get_available_classes()
                    st.write("Available classes:", available_classes)
    with tab3:
        st.header("Settings")
        available_classes = get_available_classes()
        st.write("Available tool classes:", available_classes)

# Quantization part
def quantize_model(model, dummy_input):
    """Quantize the PyTorch model"""
    model.eval()
    model.fuse_model()
    model_prepared = torch.quantization.prepare(model)
    model_prepared(dummy_input)
    model_quantized = torch.quantization.convert(model_prepared)
    return model_quantized

# Convert PyTorch Model to ONNX
def convert_to_onnx(model, dummy_input, onnx_path):
    """Convert the PyTorch model to ONNX format"""
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11)

# Optimize the Quantized Model with Hailo Dataflow Compiler
def optimize_with_hailo(onnx_path, hef_path):
    """Optimize the quantized model with Hailo Dataflow Compiler"""
    os.system(f"hailo_compile {onnx_path} --output {hef_path}")

# Evaluate the Quantized Model Using Hailo Emulator
def evaluate_with_hailo_emulator(hef_path):
    """Evaluate the quantized model using Hailo Emulator"""
    from hailo_sdk import InferRunner
    infer_runner = InferRunner(hef_path)
    input_data = ...  # Prepare your input data
    output = infer_runner.infer(input_data)
    print(output)

if __name__ == "__main__":
    main()

    # Quantization and optimization process
    model = load_feature_extractor()
    dummy_input = torch.randn(1, 3, 224, 224)
    model_quantized = quantize_model(model, dummy_input)
    onnx_path = "model_quantized.onnx"
    hef_path = "model_quantized.hef"
    convert_to_onnx(model_quantized, dummy_input, onnx_path)
    optimize_with_hailo(onnx_path, hef_path)
    evaluate_with_hailo_emulator(hef_path)
