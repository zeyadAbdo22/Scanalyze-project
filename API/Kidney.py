from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from similarity import check_similarity
from utils import preprocess_image  # moved to utils
from PIL import Image
from io import BytesIO
import numpy as np
import logging
import json

# Setup logging
logger = logging.getLogger("kidney-ultrasound")
logger.setLevel(logging.INFO)

# FastAPI router for kidney prediction
router = APIRouter()

# Kidney class labels
KIDNEY_LABELS = ["Cyst", "Normal", "Stone", "Tumor"]

@router.get("/")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"message": "KIDNEY_ Detection API is running"}

@router.post("/predict")
async def predict_kidney_image(request: Request, file: UploadFile = File(...)):
    """
    Predicts kidney condition from ultrasound image.

    Args:
        request (Request): FastAPI request with model state.
        file (UploadFile): Uploaded ultrasound image.

    Returns:
        dict: Prediction result and confidence.
    """
    try:
        logger.info(f"Received kidney image: {file.filename}")

        # Read uploaded image
        image_bytes = await file.read()

        # Step 1: Validate image as medical using similarity model
        similarity_response = check_similarity(image_bytes)
        similarity_result = json.loads(similarity_response.body.decode())

        logger.info(f"Similarity check result: {similarity_result}")

        if similarity_result.get("prediction") == "not-medical":
            raise HTTPException(status_code=400, detail="File is not a valid medical image.")

        # Step 2: Preprocess image
        image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_array = preprocess_image(image_pil, model_type="kidney")

        # Step 3: Load kidney model from app state
        kidney_model = request.app.state.kidney_model
        if kidney_model is None:
            raise ValueError("Kidney model not loaded.")

        # Step 4: Predict
        prediction = kidney_model.predict(image_array)
        predicted_index = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))

        result_label = KIDNEY_LABELS[predicted_index]
        logger.info(f"Prediction: {result_label} | Confidence: {confidence:.2f}")

        return {
            "success": True,
            "filename": file.filename,
            "prediction": result_label,
            "confidence": confidence
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
