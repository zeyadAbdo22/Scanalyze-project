from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from utils import preprocess_image
from similarity import check_similarity

from PIL import Image
from io import BytesIO
import numpy as np
import logging
import json

# Configure logging
logger = logging.getLogger("lung-xray")
logger.setLevel(logging.INFO)

# Initialize FastAPI router
router = APIRouter()

# Labels used by the lung x-ray classification model
CLASS_LABELS = {
    0: "Covid",
    1: "Normal",
    2: "Pneumonia",
    3: "Tuberculosis"
}


@router.get("/")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "message": "Chest X-ray API is up and running"
    }


@router.post("/predict")
async def predict_lung_condition(request: Request, file: UploadFile = File(...)):
    """
    Predicts the lung condition (Covid, Pneumonia, etc.) from chest X-ray image.

    Args:
        request (Request): FastAPI request with model in app state.
        file (UploadFile): Uploaded X-ray image.

    Returns:
        dict: Prediction results with class label and confidence.
    """
    try:
        logger.info(f"Received prediction request for file: {file.filename}")

        # Ensure model is loaded in app state
        model = request.app.state.lung_xray_model
        if model is None:
            raise ValueError("Lung X-ray model not initialized.")

        # Check if uploaded file is an image
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

        # Read image bytes
        image_bytes = await file.read()

        # Similarity check to validate if it's a medical image
        similarity_response = check_similarity(image_bytes)
        similarity_result = json.loads(similarity_response.body.decode())

        logger.info(f"Similarity check result: {similarity_result}")

        if similarity_result.get("prediction") == "not-medical":
            raise HTTPException(status_code=400, detail="Uploaded file is not a medical image.")

        # Convert image bytes to PIL image and preprocess
        image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
        preprocessed_image = preprocess_image(image_pil)

        # Run prediction
        prediction = model.predict(preprocessed_image, verbose=0)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        result_label = CLASS_LABELS[predicted_class]
        logger.info(f"Prediction result: {result_label}, Confidence: {confidence:.2f}")

        return {
            "success": True,
            "filename": file.filename,
            "prediction": result_label,
            "confidence": confidence
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
