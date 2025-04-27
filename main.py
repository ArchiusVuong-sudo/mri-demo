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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Roboflow
@st.cache_resource
def load_model():
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace().project("brain-tumour-detection-mri")
    return project.version(1).model

# Function to resize image to fixed dimensions
def resize_image(image, target_size=(400, 400)):
    # Handle numpy array (from OpenCV)
    if isinstance(image, np.ndarray):
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    # Handle PIL Image
    elif isinstance(image, Image.Image):
        return image.resize(target_size, Image.Resampling.LANCZOS)
    else:
        raise TypeError("Image must be either numpy array or PIL Image")

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
    # Sidebar
    with st.sidebar:
        st.title("🧠 Brain Tumor Detection")
        st.markdown("---")
        st.markdown("### About")
        st.info("This application uses AI to detect brain tumors in MRI scans. Upload an image or select from our catalog to begin analysis.")
        st.markdown("---")
        st.markdown("### Instructions")
        st.markdown("1. Upload an MRI image or select from catalog")
        st.markdown("2. Click 'Analyze' to process the image")
        st.markdown("3. View detection results and details")
        st.markdown("---")

    # Main content
    st.title("Brain Tumor Detection in MRI Scans")
    
    # Load model
    model = load_model()
    
    # Initialize session state for results
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'annotated_image' not in st.session_state:
        st.session_state.annotated_image = None

    # Create two main columns for input and results
    input_col, results_col = st.columns([1, 1])

    # Input section (left column)
    with input_col:
        st.header("📤 Input Options")
        
        # Upload section
        st.subheader("Upload New Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.session_state.original_image = image
            
            # Resize uploaded image to fixed dimensions
            resized_upload = resize_image(image, (600, 600))
            st.image(resized_upload, caption="Uploaded Image")
            if st.button("🔍 Analyze Uploaded Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    annotated_image, result = process_image(image, model)
                    st.session_state.annotated_image = annotated_image
                    st.session_state.results = result
        
        # Catalog section
        st.markdown("---")
        st.subheader("📚 Sample MRI Images")
        catalog_images = get_catalog_images()
        
        if not catalog_images:
            st.warning("No images in the catalog. Please add images to the 'catalog_images' directory.")
        else:
            # Display catalog images in a grid with fixed dimensions
            for i in range(0, len(catalog_images), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(catalog_images):
                        img_name = catalog_images[i + j]
                        img_path = os.path.join("catalog_images", img_name)
                        img = Image.open(img_path)
                        
                        # Resize image to fixed dimensions
                        resized_img = resize_image(img, (400, 400))
                        
                        with col:
                            st.image(resized_img, caption=img_name)
                            if st.button(f"🔍 Analyze {img_name}", key=f"analyze_{img_name}", use_container_width=True):
                                with st.spinner("Analyzing image..."):
                                    annotated_image, result = process_image(img, model)
                                    st.session_state.original_image = img
                                    st.session_state.annotated_image = annotated_image
                                    st.session_state.results = result

    # Results section (right column)
    with results_col:
        st.header("📊 Detection Results")
        
        if st.session_state.annotated_image is not None:
            # Resize and display the detected image with fixed dimensions
            resized_result = resize_image(st.session_state.annotated_image, (600, 600))
            st.image(resized_result)
            
            # Display detection information in an expander
            if st.session_state.results and st.session_state.results["predictions"]:
                with st.expander("🔍 Detection Details", expanded=True):
                    for pred in st.session_state.results["predictions"]:
                        confidence = pred['confidence'] * 100
                        st.metric(
                            label=f"Tumor Detection ({pred['class']})",
                            value=f"{confidence:.1f}%",
                            delta=f"{confidence - 40:.1f}% above threshold" if confidence > 40 else None
                        )
            else:
                st.info("No tumors detected in the image.")
        else:
            st.info("Upload an image or select from the catalog to see detection results.")

if __name__ == "__main__":
    main()
