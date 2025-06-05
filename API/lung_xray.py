from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from utils import preprocess_image
from similarity import check_similarity
from PIL import Image
from io import BytesIO
import numpy as np
import logging
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Class labels
CLASS_LABELS = {0: "could be Covid",
                1: "Normal",
                2: "could be Pneumonia", 
                3: "could be Tuberculosis"}


@router.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Chest X-ray API is up and running"}


@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        logger.info(f"Received prediction request for file: {file.filename}")

        # Get model from app state
        model = request.app.state.lung_xray_model
        if model is None:
            raise ValueError("Model not initialized")

        # Validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        contents = await file.read()

        # Parse similarity check response
        similar = check_similarity(contents)
        similarity_data = similar.body.decode()  # Convert bytes to string
        similarity_result = json.loads(similarity_data)  # Parse JSON string

        if similarity_result.get("prediction") == "not-medical":
            raise HTTPException(
                status_code=400,
                detail="File is not a valid medical image"
            )

        # Add logging for debugging
        logger.info(f"Similarity check result: {similarity_data}")


        img = Image.open(BytesIO(contents)).convert("RGB")
        img_array = preprocess_image(img,model_type="resnet")

        # Predict
        prediction = model.predict(img_array, verbose=0)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))


        result = CLASS_LABELS[predicted_class]
        logger.info(f"Prediction result: {result}")

        return {
            "success": True,
            "filename": file.filename,
            "prediction": result,
            "confidence": confidence
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
