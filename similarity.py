from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi import APIRouter
import numpy as np
from PIL import Image
import io

from utils import load_model_from_azure  # Import model loader from utils

# Load the trained model from Azure Blob Storage
model = load_model_from_azure("https://scanalyzestorage.blob.core.windows.net/loadmodel/medical_scan_checker.h5")

def prepare_image(img_bytes):
    """
    Preprocesses the uploaded image:
    - Converts to RGB
    - Resizes to 224x224
    - Normalizes pixel values to [0, 1]
    - Adds batch dimension
    """
    img = Image.open(io.BytesIO(img_bytes))   # Load image from bytes
    img = img.convert('RGB')                  # Ensure RGB mode

    if img.size != (224, 224):
        img = img.resize((224, 224))          # Resize if not already 224x224

    img = np.array(img) / 255.0               # Normalize pixel values
    img = np.expand_dims(img, axis=0)         # Add batch dimension
    return img

def check_similarity(file):
    """
    Predicts whether the uploaded image is a medical scan.
    Returns prediction result and confidence score.
    """
    img = prepare_image(file)                 # Preprocess the input image

    prediction = model.predict(img)           # Get model prediction
    confidence = float(np.max(prediction))    # Extract max confidence

    # Determine label based on threshold
    result = "medical" if prediction[0] > 0.5 else "not-medical"

    # Respond with prediction and confidence score
    return JSONResponse(content={
        "prediction": result,
        "confidence": confidence
    })
