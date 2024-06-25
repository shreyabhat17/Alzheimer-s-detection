from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = load_model('my_model.h5')

def predict(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    return class_id

@app.route('/')
def index():
    # Add the following line to initialize the prediction variable
    prediction = None
    return render_template('index.html', prediction=prediction)


@app.route('/predict', methods=['POST'])
def predict_route():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file found!")

        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        class_id = predict(file_path)
        class_labels = {0: 'Mild Dementia', 1: 'Moderate Dementia', 2: 'Non Dementia', 3: 'Very Mild Dementia'}
        prediction = class_labels[class_id]

        # Pass the prediction variable to the template
        return render_template('index.html', prediction=prediction, image_file=file_path)


if __name__ == '__main__':
    app.run(debug=True)
