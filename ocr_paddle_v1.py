import streamlit as st
import cv2
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import pandas as pd

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # You can change 'en' to other languages like 'ch', 'fr', etc.

# Function to preprocess the image
def preprocess_image(image):
    # Ensure image has at least 3 dimensions before conversion
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # Already grayscale
    gray = cv2.medianBlur(gray, 3)
    return gray

# Function to perform OCR
def perform_ocr(image):
    result = ocr.ocr(image, cls=True)
    return result

# Function to detect and decode QR codes using OpenCV
def detect_qr_codes(image):
    qr_detector = cv2.QRCodeDetector()
    decoded_text, points, _ = qr_detector.detectAndDecode(image)
    return decoded_text, points

# Function to detect barcodes using OpenCV
def detect_barcodes(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # Already grayscale
    
    grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(grad_x, grad_y)
    gradient = cv2.convertScaleAbs(gradient)
    blurred = cv2.blur(gradient, (3, 3))
    _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    barcodes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 2 <= aspect_ratio <= 5 and w > 80 and h > 20:  # Added size constraints
            barcode_region = image[y:y + h, x:x + w]
            # Convert to proper format for PaddleOCR
            if isinstance(barcode_region, np.ndarray):
                barcode_text = ocr.ocr(barcode_region, cls=True)
                if barcode_text and len(barcode_text) > 0 and len(barcode_text[0]) > 0:
                    for line in barcode_text:
                        for word_info in line:
                            if word_info[1][0]:  # Check if text is detected
                                barcodes.append(word_info[1][0])
    return barcodes

# Function to cross-check extracted text with inventory database
def cross_check_inventory(text, inventory_df):
    return text in inventory_df['SerialNumber'].values

# Streamlit app
st.title("Comprehensive OCR Web App")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
detection_type = st.selectbox("Select Detection Type", ["QR Code", "Barcode", "Serial Number"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    preprocessed_image = preprocess_image(image)
    
    if detection_type == "QR Code":
        decoded_text, points = detect_qr_codes(image)
        st.subheader("QR Code Result")
        st.write(f"Decoded Text: {decoded_text}" if decoded_text else "No QR code detected.")
    
    elif detection_type == "Barcode":
        barcodes = detect_barcodes(image)
        st.subheader("Barcode Result")
        if barcodes:
            for barcode in barcodes:
                st.write(f"Barcode Text: {barcode}")
        else:
            st.write("No barcodes detected. Try adjusting the image or improving lighting.")
    
    elif detection_type == "Serial Number":
        ocr_result = perform_ocr(image)  # Using original image for better OCR
        st.subheader("OCR Result")
        
        detected_texts = []
        if ocr_result:
            for line in ocr_result:
                for word_info in line:
                    if word_info[1][0]:  # Text content is in [1][0]
                        detected_text = word_info[1][0]
                        detected_texts.append(detected_text)
                        st.write(f"Detected text: {detected_text}")
        
        if not detected_texts:
            st.write("No text detected. Try adjusting the image quality or contrast.")
        
        # Inventory check
        inventory_data = {'SerialNumber': ['ABC123', 'DEF456', 'GHI789']}
        inventory_df = pd.DataFrame(inventory_data)
        
        st.subheader("Inventory Cross-Check")
        found_match = False
        for text in detected_texts:
            if cross_check_inventory(text, inventory_df):
                st.write(f"Match found: '{text}' is in inventory.")
                found_match = True
        
        if not found_match:
            st.write("No matching serial numbers found in inventory.")
    
    # Display annotated image (optional)
    # For demonstration, just showing the original image again
    # In a full implementation,I would draw boxes around detected elements
    st.image(image, caption='Processed Image', use_column_width=True)