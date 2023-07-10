import streamlit as st
import os
import cv2
import shutil
from ultralytics import YOLO

def main():
    st.title("Image Upload and Detection")

    # Display the upload file dialog
    uploaded_file = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Remove the existing folder if it exists
        folder_path = 'runs'
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

        # Save the uploaded image to a folder
        with open(os.path.join('static/uploads', uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Run YOLO detection
        model = YOLO('best.pt')
        model.predict('static/uploads/' + image.filename, save=True)
        # Load the detected image
        img_path = os.path.join('runs/detect/yolov8n', uploaded_file.name)
        img = cv2.imread(img_path)
        result_path = os.path.join('static/results', uploaded_file.name)
        cv2.imwrite(result_path, img)

        # Display the detected image
        st.image(result_path, use_column_width=True)


if __name__ == '__main__':
    main()
