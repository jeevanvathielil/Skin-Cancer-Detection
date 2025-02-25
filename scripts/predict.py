import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load model
model = tf.keras.models.load_model('../models/skin_cancer_detector.h5')

def predict_image(image_path):
    img = np.array(Image.open(image_path).resize((224, 224))) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]
    class_label = "Malignant" if prediction >= 0.5 else "Benign"

    plt.imshow(Image.open(image_path))
    plt.title(f'Prediction: {class_label}')
    plt.axis('off')
    plt.show()

predict_image("test_image.jpg")
