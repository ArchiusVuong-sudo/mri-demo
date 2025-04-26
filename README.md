# Brain Tumor Detection MRI

A Streamlit application for detecting brain tumors in MRI images using the Roboflow API.

## Features

- Upload your own MRI images for analysis
- Select from a catalog of pre-loaded MRI images
- Visualize detection results with bounding boxes and labels
- View confidence scores for each detection

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run main.py
   ```
2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)
3. Use the "Upload Image" tab to upload your own MRI images
4. Use the "Catalog Images" tab to select from pre-loaded images

## Adding Images to the Catalog

To add images to the catalog:
1. Place your MRI images in the `catalog_images` directory
2. Supported formats: JPG, JPEG, PNG

## Requirements

- Python 3.7+
- Streamlit
- Roboflow
- OpenCV
- NumPy
- Pillow
- Supervision

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
