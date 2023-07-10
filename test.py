from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import io
import base64
from ultralytics import YOLO

app = Flask(__name__)

# Load the pretrained YOLOv8n model
model = YOLO('best.pt')

# Custom Jinja2 filter to encode data in base64


def base64_encode(data):
    if data is None:
        return ""
    return base64.b64encode(data).decode('utf-8')


# Register the filter in the Flask app
app.jinja_env.filters['base64_encode'] = base64_encode


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded image file
    image = request.files['image']

    # Open the image file using PIL
    img = Image.open(image)

    # Convert PIL image to numpy array
    source = np.array(img)

    # Run inference on the source using the YOLO model
    results = model(source)

    # Retrieve the original image from the results
    orig_img = results.imgs[0]

    # Convert the original image to bytes
    img_bytes = io.BytesIO()
    orig_img.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()

    return render_template('display.html', uploaded_image_data=img_bytes)


if __name__ == '__main__':
    app.run()
