from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from utils import preprocess_image
from similarity import check_similarity

import logging
import json
from io import BytesIO
from PIL import Image

# Setup logger
logger = logging.getLogger("brain-tumor")
logger.setLevel(logging.INFO)

# Initialize FastAPI router
router = APIRouter()

# Class labels for binary classification
CLASS_LABELS = {
    0: "might have a brain tumor",
    1: "Healthy"
}


@router.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Brain Tumor Detection API is running"
    }


@router.post("/predict")
async def predict_brain_tumor(request: Request, file: UploadFile = File(...)):
    """
    Predicts brain tumor status from an uploaded image file.
    
    Args:
        request (Request): FastAPI request containing app state.
        file (UploadFile): The uploaded image file.
    
    Returns:
        dict: A dictionary with prediction results.
    """
    try:
        logger.info(f"Received prediction request for: {file.filename}")

        # Validate model availability
        brain_model = request.app.state.brain_model
        if brain_model is None:
            raise ValueError("Brain tumor model is not loaded.")

        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image.")

        # Read image content
        image_bytes = await file.read()

        # Run similarity check to confirm it's a medical image
        similarity_response = check_similarity(image_bytes)
        response_body = similarity_response.body.decode()
        similarity_result = json.loads(response_body)

        logger.info(f"Similarity check result: {similarity_result}")

        if similarity_result.get("prediction") == "not-medical":
            raise HTTPException(status_code=400, detail="Not a valid medical image.")

        # Open and preprocess image
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        preprocessed = preprocess_image(img)

        # Run prediction
        prediction = brain_model.predict(preprocessed, verbose=0)
        predicted_class = int(prediction[0][0] > 0.5)
        confidence_score = float(prediction[0][0])

        prediction_label = CLASS_LABELS[predicted_class]
        logger.info(f"Prediction: {prediction_label} (confidence: {confidence_score:.2f})")

        return {
            "success": True,
            "filename": file.filename,
            "prediction": prediction_label,
            "confidence": confidence_score
        }

    except HTTPException:
        raise  # Allow FastAPI to handle HTTPExceptions directly
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
