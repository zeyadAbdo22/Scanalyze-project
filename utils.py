import os
import requests
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras import backend as K

def load_model_from_azure(model_url: str):
    """
    Downloads a Keras model from Azure Blob Storage and loads it.
    Disables GPU and oneDNN for compatibility.
    """
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization

        response = requests.get(model_url, stream=True)
        if response.status_code == 200:
            model_file_path = "/tmp/model.h5"
            with open(model_file_path, "wb") as model_file:
                for chunk in response.iter_content(chunk_size=8192):
                    model_file.write(chunk)

            model = load_model(model_file_path, compile=False)
            print(f"Model loaded successfully from {model_url}")
            return model
        else:
            raise Exception(f"Failed to download model from {model_url} (Status code: {response.status_code})")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def preprocess_image(img, model_type="resnet"):
    """
    Prepares an image for either ResNet50 or MobileNetV2:
    - Resizes to 224x224
    - Converts to array and adds batch dimension
    - Applies model-specific preprocessing
    """
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    if model_type.lower() == "mobilenet":
        img_array = mobilenet_preprocess(img_array)
    else:
        img_array = resnet_preprocess(img_array)

    return img_array
