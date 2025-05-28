from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
from fastapi import APIRouter
from utils import load_model_from_azure

# Load the pre-trained model from Azure Blob Storage
model = load_model_from_azure("https://scanalyzestorage.blob.core.windows.net/loadmodel/medical_scan_checker.h5")


def prepare_image(img_bytes):
    # Open the image from raw bytes
    img = Image.open(io.BytesIO(img_bytes))
    # Convert image to RGB format (in case it's grayscale or other)
    img = img.convert('RGB')

    # Resize image to 224x224 if it doesn't match target size
    if img.size != (224, 224):
        img = img.resize((224, 224))

    # Convert image to numpy array
    img = np.array(img)
    # Normalize pixel values to [0, 1]
    img = img / 255.0

    # Add batch dimension for model input shape (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)
    return img


def check_similarity(file):
    # Prepare the image for model prediction
    img = prepare_image(file)

    # Get model prediction probabilities
    prediction = model.predict(img)
    # Extract highest confidence score
    confidence = np.max(prediction)

    # Classify as 'medical' if prediction output > 0.5, else 'not-medical'
    result = "medical" if prediction[0] > 0.5 else "not-medical"

    # Confidence threshold check for 'medical' class >= 0.95 included (though response is same either way)
    if result == "medical" and confidence >= 0.95:
        return JSONResponse(content={
            "prediction": result,
            "confidence": float(confidence)
        })
    else:
        return JSONResponse(content={
            "prediction": result,
            "confidence": float(confidence)
        })
