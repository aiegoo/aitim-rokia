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
    # Convert to numpy array if it's a PIL image
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to grayscale if it's RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply Canny edge detection
    edges = cv2.Canny(enhanced, 50, 150)

    # Convert back to PIL Image
    return Image.fromarray(edges)


def preprocess_adaptive_threshold(image):
    """Preprocessing with adaptive thresholding"""
    # Convert to numpy array if it's a PIL image
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to grayscale if it's RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Denoise
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Convert back to PIL Image
    return Image.fromarray(thresh)


def preprocess_sobel_filter(image):
    """Preprocessing with Sobel filter for edge detection"""
    # Convert to numpy array if it's a PIL image
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to grayscale if it's RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply Sobel operator
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the magnitude of gradient
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Normalize to 0-255
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Convert back to PIL Image
    return Image.fromarray(magnitude)


def preprocess_morphological(image):
    """Preprocessing with morphological operations"""
    # Convert to numpy array if it's a PIL image
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to grayscale if it's RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply threshold
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply morphological operations
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Convert back to PIL Image
    return Image.fromarray(opening)


def preprocess_combined_edges(image):
    """Preprocessing combining multiple edge detection methods"""
    # Convert to numpy array if it's a PIL image
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to grayscale if it's RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply Canny edge detection
    edges_canny = cv2.Canny(enhanced, 30, 150)

    # Apply Laplacian edge detection
    laplacian = cv2.Laplacian(enhanced, cv2.CV_8U)

    # Combine edge detection methods
    combined = cv2.bitwise_or(edges_canny, laplacian)

    # Convert back to PIL Image
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
    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Feature extractor class using pre-trained EfficientNet
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Load pre-trained EfficientNet
        self.model = models.efficientnet_b0(weights='DEFAULT')  # Updated from pretrained=True
        # Remove the classification head
        self.features = nn.Sequential(*list(self.model.children())[:-1])
        # Add pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Freeze all parameters to prevent training
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        # L2 normalize features for cosine similarity
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
    # Check if image is grayscale or RGB
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

    # Extract features
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
    # Create directory if it doesn't exist
    tool_dir = os.path.join(REFERENCE_PATH, tool_name)
    os.makedirs(tool_dir, exist_ok=True)

    # Base filename
    base_filename = str(int(time.time()))

    # Process with each preprocessing function and save features
    feature_count = 0
    for i, preprocess_func in enumerate(PREPROCESSING_FUNCTIONS):
        try:
            # Process the image
            processed_image = preprocess_func(image)

            # Extract features from the processed image
            features = extract_features(model, processed_image)

            # Save features and processed image
            torch.save(features, os.path.join(tool_dir, f"{base_filename}_aug{i}.pt"))
            processed_image.save(os.path.join(tool_dir, f"{base_filename}_aug{i}.jpg"))

            feature_count += 1
        except Exception as e:
            st.error(f"Error processing with augmentation {i}: {e}")
            continue

    # Update global CLASSES list
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

    # Update global CLASSES list
    CLASSES = classes
    return classes


# Function to scan a frame with multiple augmentations and ensemble voting
def scan_frame_ensemble(model, frame):
    """Scan a frame with multiple preprocessing techniques and ensemble voting"""
    # Force update class list to ensure we're using only classes with reference data
    available_classes = get_available_classes()

    # If no classes are available, return early
    if not available_classes:
        return "No reference data available", [], 0, []

    # Results from each preprocessor
    augmentation_results = []
    processed_images = []

    # Apply each preprocessing function
    for preprocess_func in PREPROCESSING_FUNCTIONS:
        try:
            # Process the image
            processed_image = preprocess_func(frame)
            processed_images.append(processed_image)

            # Extract features
            features = extract_features(model, processed_image)

            # Compare with reference features
            similarities = []

            for tool_name in available_classes:
                tool_dir = os.path.join(REFERENCE_PATH, tool_name)
                if os.path.exists(tool_dir):
                    tool_similarities = []  # Store similarities for this tool

                    for filename in os.listdir(tool_dir):
                        if filename.endswith('.pt'):  # Feature files
                            try:
                                feature_path = os.path.join(tool_dir, filename)
                                ref_features = torch.load(feature_path)
                                similarity = compute_similarity(features, ref_features)
                                tool_similarities.append((tool_name, similarity, filename))
                            except Exception as e:
                                continue

                    # Only keep the highest similarity score for this tool
                    if tool_similarities:
                        # Sort by similarity (higher is better)
                        tool_similarities.sort(key=lambda x: x[1], reverse=True)
                        # Add the best match for this tool
                        similarities.append(tool_similarities[0])

            # If similarities found
            if similarities:
                # Sort by similarity (higher is better)
                similarities.sort(key=lambda x: x[1], reverse=True)

                # Get top match
                top_match = similarities[0]
                result = top_match[0]
                confidence = top_match[1]

                # Store result
                augmentation_results.append((result, confidence, similarities[:3]))

        except Exception as e:
            continue

    # If no augmentation produced results
    if not augmentation_results:
        return "Unknown Tool", [], 0, processed_images

    # Combine results from all augmentations using weighted voting
    vote_results = {}
    for result, confidence, _ in augmentation_results:
        if result in vote_results:
            vote_results[result] += confidence  # Weight votes by confidence
        else:
            vote_results[result] = confidence

    # Get the result with highest weighted votes
    if vote_results:
        ensemble_result = max(vote_results.items(), key=lambda x: x[1])
        final_result = ensemble_result[0]

        # Calculate average confidence for the winning class
        matching_confidences = [conf for res, conf, _ in augmentation_results if res == final_result]
        avg_confidence = sum(matching_confidences) / len(matching_confidences) if matching_confidences else 0

        # Get top matches from all augmentations for the winning class
        top_similarities = []
        for _, _, similarities in augmentation_results:
            top_similarities.extend([s for s in similarities if s[0] == final_result])

        # Sort and get top 5
        top_similarities.sort(key=lambda x: x[1], reverse=True)
        top_5 = top_similarities[:5] if top_similarities else []

        return final_result, top_5, avg_confidence, processed_images
    else:
        return "Unknown Tool", [], 0, processed_images


# Main Streamlit UI with enhanced live video streaming
def main():
    st.title("Tool Recognition System")


    # Force load available classes on startup
    load_available_classes()

    # Load feature extractor
    model = load_feature_extractor()

    # Ensure reference directory exists
    if not os.path.exists(REFERENCE_PATH):
        os.makedirs(REFERENCE_PATH)

    # Get available classes
    available_classes = get_available_classes()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Live Stream", "Add Reference Tool", "Settings"])

    with tab1:


        # Add warning if no reference data is available
        if not available_classes:
            st.warning("No reference data available. Please add reference tools first.")

        # Setup live streaming
        run_livestream = st.checkbox("Start Live Stream")
        camera_id =0



        # Container for the video feed
        stream_placeholder = st.empty()
        confidence_threshold=0.35

        if run_livestream:
            with st.spinner("Starting live stream..."):
                # Open camera
                cap = cv2.VideoCapture(int(camera_id))

                # Check if camera opened successfully
                if not cap.isOpened():
                    st.error("Unable to open camera. Please check camera ID.")
                    run_livestream = False
                else:
                    # Set frame width and height
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                    # Variables for processing
                    frame_count = 1
                    process_interval = 1
                    last_result = "Waiting for detection..."

                    # For sliding window of results (for stability)
                    result_window = []
                    window_size = 1  # Increased window size for better stability

                    while run_livestream:
                        # Force refresh available classes periodically
                        if frame_count % 100 == 0:
                            available_classes = get_available_classes()

                        ret, frame = cap.read()
                        if not ret:
                            st.error("Error reading from camera.")
                            break

                        # Convert frame to RGB for display
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Create copy for annotation
                        annotated_frame = frame_rgb.copy()

                        # Process frames according to interval
                        if frame_count % process_interval == 0:
                            # Convert frame to PIL Image
                            pil_frame = Image.fromarray(frame_rgb)

                            # Process frame with ensemble of techniques
                            result, top_5, confidence, _ = scan_frame_ensemble(model, pil_frame)

                            # Apply the user-defined confidence threshold
                            effective_result = result if confidence > confidence_threshold else "Unknown Tool"

                            # Add to sliding window
                            result_window.append((effective_result, confidence))
                            # Keep window at desired size
                            if len(result_window) > window_size:
                                result_window.pop(0)

                            # Update result based on window
                            if result_window:
                                # Filter results by confidence threshold
                                valid_results = [(r, c) for r, c in result_window if
                                                 r != "Unknown Tool" and c > confidence_threshold]

                                if valid_results:
                                    # Count occurrences of each result
                                    result_counts = Counter([r for r, _ in valid_results])
                                    # Get most common result
                                    most_common = result_counts.most_common(1)[0]

                                    last_result = most_common[0]

                                else:
                                    last_result = "Unknown Tool"

                            # Add a semi-transparent background for text visibility
                            overlay = annotated_frame.copy()
                            # Create a black background at the top of the frame
                            cv2.rectangle(overlay, (0, 0), (annotated_frame.shape[1], 60), (0, 0, 0), -1)
                            # Blend with original frame
                            cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)

                            # Set text color based on detection (green for known, red for unknown)
                            text_color = (0, 255, 0) if last_result != "Unknown Tool" else (255, 0, 0)

                            # Add text with current detection in center of the black bar
                            text = last_result
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                            text_x = (annotated_frame.shape[1] - text_size[0]) // 2
                            cv2.putText(
                                annotated_frame,
                                text,
                                (text_x, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                text_color,
                                2
                            )

                        # Display frame
                        with stream_placeholder.container():
                            st.image(annotated_frame, use_column_width=True)

                        frame_count += 1
                        time.sleep(0.03)  # Small delay to reduce CPU usage

                        # Check if checkbox is still checked
                        if not st.session_state.get("run_livestream", run_livestream):
                            break

                    # Release camera
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

                    # Force refresh available classes
                    available_classes = get_available_classes()
                    st.write("Available classes:", available_classes)

    with tab3:
        st.header("Settings")

        # Get current available classes
        available_classes = get_available_classes()
        st.write("Available tool classes:", available_classes)




if __name__ == "__main__":
    main()