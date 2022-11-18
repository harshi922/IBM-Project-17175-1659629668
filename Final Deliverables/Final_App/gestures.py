import keras
import cv2
import numpy as np
import os
reconstructed_model = keras.models.load_model("new_train_model.h5")

def gesture_prediction(image):
    image = cv2.imread(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, [100, 120])
    gray_image = gray_image.reshape(1, 100, 120, 1)
    prediction = reconstructed_model.predict(gray_image)
    predicted_class = np.argmax(prediction)
    if predicted_class == 0:
        return "Blank"
    elif predicted_class == 1:
        return "OK"
    elif predicted_class == 2:
        return "Thumbs Up"
    elif predicted_class == 3:
        return "Thumbs Down"
    elif predicted_class == 4:
        return "Punch"
    elif predicted_class == 5:
        return "High Five"
