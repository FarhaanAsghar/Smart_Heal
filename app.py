import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO

# Function for model prediction
def model_prediction(image):
    # Load the YOLO model
    my_new_model = YOLO('last.pt')  # Update the path to your model weights
    # Preprocessing the image
    new_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    new_image = cv2.resize(new_image, (512, 512))
    # Prediction
    new_results = my_new_model.predict(new_image)

    # Extract the original image
    orig_img = new_results[0].orig_img
    extracted_masks = new_results[0].masks.data
    masks_array = extracted_masks.cpu().numpy()

    # Set up the Matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))

    for idx, mask_image1 in enumerate(masks_array):
        mask_image = mask_image1.astype(np.uint8)
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_image)

        # Draw the bounding box and annotate
        box = new_results[0].boxes.data[idx]
        x, y, w, h = box[:4]  # Extract box coordinates
        width = abs(w - x) / 100
        length = abs(h - y) / 100
        ax.plot([x, w, w, x, x], [y, y, h, h, y], 'r-', linewidth=2, alpha=0.6)  # Draw box
        ax.text(x, y - 10, f"Area: {(length * width):.2f} cm²", color='blue', fontsize=10, backgroundcolor='white')
        ax.text(x, y - 30, f"L: {length:.2f} cm, W: {width:.2f} cm", color='blue', fontsize=10, backgroundcolor='white')

    ax.axis('off')  # Turn off axis labels
    return fig

# Streamlit App
st.title("Smart Heel")
st.write("Wound Segmentation and Measurement")
st.write("Upload an image, and the app will predict the wound area, length, and width.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Perform prediction and display results
    result_fig = model_prediction(image)
    
    # Convert the Matplotlib figure to an image and display it in Streamlit
    buf = BytesIO()
    result_fig.savefig(buf, format="png")
    buf.seek(0)
    st.image(buf, caption="Segmentation Results", use_column_width=True)
