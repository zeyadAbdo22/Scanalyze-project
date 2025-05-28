import logging
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from PIL import Image
from io import BytesIO
import numpy as np

from utils import preprocess_image  # Image preprocessing helper function

# Configure logger
logger = logging.getLogger("diabetic-retinopathy-prediction")
logger.setLevel(logging.INFO)

# Create FastAPI router
router = APIRouter()

# Define class labels
DR_CLASSES = {
    0: "Might have Diabetic Retinopathy",
    1: "Healthy",
}


@router.get("/")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"message": "Diabetic Retinopathy Detection API is running"}


@router.post("/predict")
async def predict_diabetic_retinopathy(request: Request, file: UploadFile = File(...)):
    """
    Predict diabetic retinopathy from an uploaded retinal image.

    Args:
        request (Request): FastAPI request to access app state.
        file (UploadFile): Uploaded image file.

    Returns:
        dict: Prediction result with class label and confidence score.
    """
    try:
        logger.info(f"Received prediction request for file: {file.filename}")

        # Load the model from app state
        dr_model = request.app.state.Diabetic_Retinopathy_model
        if dr_model is None:
            raise ValueError("Diabetic Retinopathy model is not loaded.")

        # Validate file type is image
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

        # Read image bytes and open with PIL
        image_bytes = await file.read()
        img = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Preprocess image for model input
        img_array = preprocess_image(img)

        # Predict using the model
        prediction = dr_model.predict(img_array, verbose=0)

        # For binary classification, threshold at 0.5
        predicted_class = int(prediction[0][0] > 0.5)
        confidence = float(prediction[0][0])

        predicted_label = DR_CLASSES[predicted_class]
        logger.info(f"Prediction completed: {predicted_label} (confidence: {confidence:.4f})")

        return {
            "success": True,
            "filename": file.filename,
            "prediction": predicted_label,
            "confidence": confidence,
        }

    except HTTPException:
        # Propagate HTTP exceptions directly
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")