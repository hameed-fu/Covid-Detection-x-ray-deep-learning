# Import Statements
import sys
import os
import glob
import re
import numpy as np

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Initializing saved Model
MODEL_PATH ='covid.h5'

# Loading Trained/Saved Model
model = load_model(MODEL_PATH)

# Funtion to make prediction on the input image
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    
    x=x/255
    x = np.expand_dims(x, axis=0)
    

    preds = model.predict_classes(x)
    
    if preds==0:
        preds="COVID-19 Detected"
    elif preds==1:
        preds="COVID-19 not Detected!"

    
    return preds

# Flask app
@app.route('/', methods=['GET'])
def index():
    # Main Web-page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        f = request.files['file']

        # To create a directory in the file system to store the upladed image
        basepath = os.path.dirname(__file__)
        if not os.path.exists('uploads'):
            os.mkdir('uploads')
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Making Prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)