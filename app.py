import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict
#hii this side vijay

# Load the YOLOv8 model
model = YOLO("animal__yolov8.pt")  # Ensure this model is in your working directory

# Streamlit UI setup
st.set_page_config(page_title="Animal Counter", layout="centered")

# Hide Streamlit menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Animal Count using YOLOv8")
st.write("Upload an image to detect and count animals in it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert to OpenCV image
    image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect and Count"):
        with st.spinner("Detecting animals..."):
            results = model(img_np)
            annotated_frame = img_np.copy()
            animal_counts = defaultdict(int)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    if confidence > 0.5:
                        class_name = model.names[cls_id]
                        animal_counts[class_name] += 1

                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = f"{class_name} ({confidence:.2f})"

                        # Draw rectangle and label
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Show results
            # st.image(annotated_frame, caption="Detection Result", channels="RGB", use_column_width=True)

            if animal_counts:
                st.success("✅ Animals Detected:")
                for animal, count in animal_counts.items():
                    st.write(f"**{animal}**: {count}")
            else:
                st.warning("No animals detected with confidence > 0.5.")
