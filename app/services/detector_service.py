import numpy as np
from PIL import Image
import tensorflow as tf


class PredictionService:
    def __init__(self, model_path: str):
        """
        Initializes the PredictionService with a pre-trained model.
        Loads the model from the specified file path.

        Args:
            model_path (str): The file path to the trained model.
        """
        # Load the pre-trained Keras model
        self.model = tf.keras.models.load_model(model_path)

    @staticmethod
    def preprocess_image(image: Image.Image):
        """
        Preprocesses the input image by resizing it to the target shape,
        normalizing, and reshaping it for model prediction.

        Args:
            image (PIL.Image.Image): The image to preprocess.

        Returns:
            np.ndarray: The preprocessed image array with the added batch dimension.
        """
        # Resize the image to 128x128 pixels to match the model input shape
        x = np.array(image.resize((128, 128)))

        # Normalize the image (if the model expects normalization like [0, 1] or [-1, 1])
        # Uncomment the line below if the model was trained with normalized values.
        # x = x / 255.0

        # Reshape the image to match the expected input shape for the model:
        # (batch size, height, width, channels), here batch size is 1.
        return x.reshape(1, 128, 128, 3)

    def predict(self, image: Image.Image):
        """
        Makes a prediction on the provided image using the pre-trained model.

        Args:
            image (PIL.Image.Image): The image to predict on.

        Returns:
            tuple: The prediction confidence (percentage) and the classification label.
        """
        # Preprocess the image before making predictions
        processed_image = self.preprocess_image(image)

        # Use the model to predict the class probabilities for the processed image
        res = self.model.predict_on_batch(processed_image)

        # Get the index of the highest probability class (classification)
        classification = np.argmax(res[0])

        # Extract the confidence (probability) of the predicted class
        confidence = res[0][classification] * 100

        return confidence, classification
