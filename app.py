import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("model/handwritten_model.h5")

# Load sample image from MNIST
mnist = tf.keras.datasets.mnist
(_, _), (test_images, test_labels) = mnist.load_data()

img = test_images[0]

# Predict
prediction = model.predict(img.reshape(1,28,28,1))

print("Predicted Digit:", np.argmax(prediction))

# Show image
plt.imshow(img, cmap='gray')
plt.title(f"Predicted: {np.argmax(prediction)}")
plt.show()
