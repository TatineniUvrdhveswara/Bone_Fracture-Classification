import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# --- Page Config ---
st.set_page_config(
    page_title="Fracture Detection System",
    page_icon="🦴",
    layout="centered"
)

# --- Title and Description ---
st.title("🦴 Fracture Detection & Classification")
st.markdown("Upload an X-ray or use your webcam to detect if a bone is fractured and classify the type.")

# --- Load Model (Cached so it doesn't reload every time) ---
@st.cache_resource
def load_model():
    # Ensure 'best.pt' is in the same folder as this script
    return YOLO("best.pt")

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'best.pt' is in the same folder.")

# --- Input Method Selection ---
option = st.radio("Choose Input Method:", ("Upload Image", "Use Webcam"))

input_image = None

# --- Logic for File Upload ---
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Convert the file to an image
        input_image = Image.open(uploaded_file)

# --- Logic for Webcam ---
elif option == "Use Webcam":
    # st.camera_input allows the user to snap a photo
    camera_file = st.camera_input("Take a picture")
    if camera_file is not None:
        input_image = Image.open(camera_file)

# --- Detection & Display ---
if input_image is not None:
    st.divider()
    col1, col2 = st.columns(2)
    
    # Show Original
    with col1:
        st.subheader("Original Input")
        st.image(input_image, use_container_width=True)

    # Perform Detection
    # Convert PIL Image to Numpy array for YOLO
    image_np = np.array(input_image)
    
    # Run the model
    results = model(image_np)
    
    # Plot the results (this draws the boxes)
    # [:, :, ::-1] converts BGR (OpenCV format) to RGB (Streamlit format)
    annotated_img = results[0].plot()[:, :, ::-1]

    # Show Result
    with col2:
        st.subheader("AI Prediction")
        st.image(annotated_img, use_container_width=True)

    # --- Textual Details ---
    st.write("### Detection Details:")
    
    # Check if any objects were detected
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            
            # Display colorful metric for each fracture found
            st.metric(label="Fracture Type", value=class_name, delta=f"{confidence:.2%} Confidence")
    else:
        st.info("No fractures detected. The bone appears healthy.")