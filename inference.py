import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load model from model folder
model = load_model("model/my_model.keras")


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))   # adjust size if trained differently
    print(img)
    img_array = image.img_to_array(img)
    print(img_array.shape)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  

    prediction = model.predict(img_array)

    if prediction.shape[-1] == 1:  # sigmoid
        label = "Dog" if prediction[0][0] > 0.5 else "Cat"
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    else:  # softmax
        class_idx = np.argmax(prediction, axis=1)[0]
        label = "Dog" if class_idx == 1 else "Cat"
        confidence = prediction[0][class_idx]

    return label, float(confidence)

if __name__ == "__main__":
    # Example: Use image from Images folder
    test_image = os.path.join("12.jpeg")  
    label, confidence = predict_image(test_image)
    print(f"Prediction: {label} (confidence: {confidence:.2f})")
