# Importing required module
from flask import Flask, render_template, request
import cv2
from ultralytics import YOLO
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded image file
    image = request.files['image']

    # Save the image to a folder
    image.save('static/uploads/' + image.filename)
    model = YOLO('best.pt')
    model.predict('static/uploads/' + image.filename,save=True)
    img = cv2.imread('runs/detect/predict/'+image.filename)
    cv2.imwrite('static/results/'+image.filename, img)
    # Define the image_name variable
    image_name = image.filename
    
    import shutil
    # Specify the path of the folder to be deleted
    folder_path = 'runs'

    # Remove the folder and its contents
    shutil.rmtree(folder_path)

    return render_template('display.html', image_name=image_name)


if __name__ == '__main__':
    app.run(debug=True)
