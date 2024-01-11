from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from data_preprocessing import preprocess
import cv2
import numpy as np

model_path = "models/cnn_model_100.h5"

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck']


def predict(model_path: str, img: str) -> str:
    """
    Predicts the label of the test image.

    Args:
        model_path: The path to the model file.
        img_path: The path to the test image.

    Returns:
        The predicted label of the test image.
    """
    # Load the model
    model = load_model(model_path)

    # Check if the image is loaded successfully
    if img is not None:
        # Resize the image to 32x32 pixels
        img_resized = cv2.resize(img, (32, 32))

        # Reshape the image for further processing
        img_reshaped = img_resized.reshape(1, 32, 32, 3)

        # Make predictions
        predictions = np.argmax(model.predict(img_reshaped))

        # Return the predicted label
        return labels[predictions]
    else:
        print(f"Error: Unable to load the image from {img_path}")


if __name__ == "__main__":
    
    img_path="bird.jpg"
    # Read the test image
    img = cv2.imread(img_path)

    predictions = predict(model_path, img)
    print(predictions)