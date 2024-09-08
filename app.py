from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load your trained model (we'll load the model when the first request is made)
model = None

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed extensions for image upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess image according to model input requirements
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128,128))  # Change target size as per your model's input shape
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, height, width, channels)
    img_array = img_array / 255.0  # Normalize pixel values if needed (0-1 range)
    return img_array

# Prediction function
def predict_sickle_cell(img_path):
    # Load the model if it hasn't been loaded yet
    global model
    if model is None:
        model = load_model('saved_model/my_model.h5')
    
    # Preprocess the image
    preprocessed_image = preprocess_image(img_path)
    
    # Make prediction
    prediction = model.predict(preprocessed_image)
    
    # Assuming binary classification, threshold at 0.5 for 'Positive' or 'Negative'
    return 'Positive' if prediction[0][0] > 0.5 else 'Negative'

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Make prediction
        prediction = predict_sickle_cell(filepath)

        # Determine result class for CSS styling
        result_class = 'positive' if prediction == 'Positive' else 'negative'
        
        return render_template('result.html', filename=filename, prediction=prediction, result_class=result_class)
    
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
