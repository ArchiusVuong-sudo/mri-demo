import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import io
from roboflow import Roboflow
import supervision as sv
from dotenv import load_dotenv

load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Detection MRI",
    page_icon="🧠",
    layout="wide"
)

# Initialize Roboflow
@st.cache_resource
def load_model():
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace().project("brain-tumour-detection-mri")
    return project.version(1).model

# Function to process image and get predictions
def process_image(image, model):
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Save image temporarily for Roboflow API
    temp_path = "temp_image.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    
    # Get predictions
    result = model.predict(temp_path, confidence=40, overlap=30).json()
    
    # Clean up temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    # Extract labels and create detections
    labels = [item["class"] for item in result["predictions"]]
    detections = sv.Detections.from_roboflow(result)
    
    # Create annotators with default settings
    label_annotator = sv.LabelAnnotator(
        text_thickness=2,
        text_scale=1,
        text_padding=3
    )
    
    box_annotator = sv.BoxAnnotator(
        thickness=3
    )
    
    # Annotate image
    annotated_image = box_annotator.annotate(
        scene=image_np, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)
    
    # Apply custom colors manually
    # Convert to BGR for OpenCV
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    
    # Draw green boxes and white text
    for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
        # Draw green box
        cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        
        # Add white text with green background
        label = labels[i] if i < len(labels) else ""
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(annotated_image, (int(x1), int(y1) - 25), (int(x1) + text_size[0], int(y1)), (0, 255, 0), -1)
        cv2.putText(annotated_image, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Convert back to RGB for display
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    return annotated_image, result

# Function to get catalog images
def get_catalog_images():
    catalog_dir = "catalog_images"
    if not os.path.exists(catalog_dir):
        os.makedirs(catalog_dir)
        # Create a placeholder image if catalog is empty
        placeholder = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Add images to catalog", (50, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(catalog_dir, "placeholder.jpg"), placeholder)
    
    return [f for f in os.listdir(catalog_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Main app
def main():
    st.title("🧠 Brain Tumor Detection MRI")
    st.write("Upload an MRI image or select from our catalog to detect brain tumors.")
    
    # Load model
    model = load_model()
    
    # Initialize session state for results
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'annotated_image' not in st.session_state:
        st.session_state.annotated_image = None
    
    # Create two columns for the layout
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        # Upload section
        st.header("Upload Your MRI Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        # Process uploaded image
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.session_state.original_image = image
            
            # Process button
            if st.button("Analyze Uploaded Image"):
                with st.spinner("Analyzing image..."):
                    # Process image
                    annotated_image, result = process_image(image, model)
                    st.session_state.annotated_image = annotated_image
                    st.session_state.results = result
        
        # Catalog section
        st.header("Catalog Images")
        
        # Get catalog images
        catalog_images = get_catalog_images()
        
        if not catalog_images:
            st.warning("No images in the catalog. Please add images to the 'catalog_images' directory.")
        else:
            # Create a grid for catalog images
            cols_per_row = 2
            for i in range(0, len(catalog_images), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(catalog_images):
                        img_name = catalog_images[i + j]
                        img_path = os.path.join("catalog_images", img_name)
                        img = Image.open(img_path)
                        
                        with col:
                            st.image(img, caption=img_name, use_column_width=True)
                            if st.button(f"Analyze {img_name}", key=f"analyze_{i+j}"):
                                with st.spinner("Analyzing image..."):
                                    # Process image
                                    annotated_image, result = process_image(img, model)
                                    
                                    # Update session state
                                    st.session_state.original_image = img
                                    st.session_state.annotated_image = annotated_image
                                    st.session_state.results = result
    
    with right_col:
        # Results section
        st.header("Detection Results")
        
        if st.session_state.original_image is not None and st.session_state.annotated_image is not None:
            # Display original and annotated images
            st.subheader("Original Image")
            st.image(st.session_state.original_image, use_column_width=True)
            
            st.subheader("Detected Image")
            st.image(st.session_state.annotated_image, use_column_width=True)
            
            # Display detection information
            if st.session_state.results and st.session_state.results["predictions"]:
                st.subheader("Detection Details")
                for pred in st.session_state.results["predictions"]:
                    st.write(f"- Class: {pred['class']}, Confidence: {pred['confidence']:.2f}")
            else:
                st.info("No tumors detected in the image.")
        else:
            st.info("Upload an image or select from the catalog to see detection results.")

if __name__ == "__main__":
    main()
