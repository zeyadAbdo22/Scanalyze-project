import os
import requests
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

def load_model_from_azure(model_url: str):
    """
    Downloads and loads a Keras model from an Azure Blob Storage URL.

    Args:
        model_url (str): The URL of the model in Azure Blob Storage.

    Returns:
        keras.Model: The loaded Keras model.
    """
    try:
        # Disable GPU and ONEDNN optimizations
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

        # Send GET request to download the model
        response = requests.get(model_url, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download model from {model_url}")

        # Save model to temporary file
        model_file_path = "/tmp/model.h5"
        with open(model_file_path, "wb") as model_file:
            for chunk in response.iter_content(chunk_size=8192):
                model_file.write(chunk)

        # Load model without compilation
        model = load_model(model_file_path, compile=False)
        print(f"Model loaded successfully from {model_url}")
        return model

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def preprocess_image(img):
    """Preprocesses the uploaded image to be compatible with ResNet50 model input."""
    # Resize image to 224x224 pixels as expected by ResNet50
    img = img.resize((224, 224))
    
    # Convert PIL Image to numpy array
    img_array = image.img_to_array(img)
    
    # Add batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply ResNet50-specific preprocessing (mean subtraction, scaling, etc.)
    img_array = preprocess_input(img_array)
    
    return img_array