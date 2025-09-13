import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import shutil
import os

# Load your trained model
MODEL_PATH = "model/my_model.keras"
model = load_model(MODEL_PATH)

# Load FastAPI app
app = FastAPI()

# Mount static files (for CSS, images, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set template folder
templates = Jinja2Templates(directory="templates")

# Prediction function
def predict(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # use same size as training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)

    if prediction.shape[-1] == 1:  # Sigmoid
        label = "Dog" if prediction[0][0] > 0.5 else "Cat"
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    else:  # Softmax
        class_idx = np.argmax(prediction, axis=1)[0]
        label = "Dog" if class_idx == 1 else "Cat"
        confidence = prediction[0][class_idx]

    return label, round(float(confidence), 2)

# Home page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# Handle image upload
@app.post("/predict", response_class=HTMLResponse)
async def predict_image(request: Request, file: UploadFile = File(...)):
    # Save uploaded image temporarily
    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Predict
    label, confidence = predict(file_path)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": f"Prediction: {label} (Confidence: {confidence:.2f})",
        "img_path": "/" + file_path
    })

