# Importing required module
from flask import Flask, render_template, request
import os
import cv2

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    import shutil
    # Specify the path of the folder to be deleted
    folder_path = 'runs'

    # Remove the folder and its contents
    shutil.rmtree(folder_path)

    # Get the uploaded image file
    image = request.files['image']

    # Save the image to a folder
    image.save('static/uploads/' + image.filename)
    os.system('yolo task=detect mode=predict model=best.pt source=static/uploads/ imgsz=640 name=yolov8n show_labels=true')
    img = cv2.imread('runs/detect/yolov8n/'+image.filename)
    cv2.imwrite('static/results/'+image.filename, img)
    # Define the image_name variable
    image_name = image.filename

    return render_template('display.html', image_name=image_name)


if __name__ == '__main__':
    app.run(debug=True)
