from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from similarity import check_similarity
from utils import preprocess_image  # نفترض إنو preprocessing نقلناها لهنا
from PIL import Image
from io import BytesIO
import numpy as np
import logging
import json

# Setup logging
logger = logging.getLogger("knee-prediction")
logger.setLevel(logging.INFO)

# FastAPI router for knee prediction
router = APIRouter()

# Knee class labels
KNEE_LABELS = ["Healthy", "Osteoporosis"]

@router.get("/")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"message": "Knee Detection API is running"}


@router.post("/predict")
async def predict_knee_image(request: Request, file: UploadFile = File(...)):
    """
    Predicts knee condition from uploaded image.

    Args:
        request (Request): FastAPI request with app state.
        file (UploadFile): Uploaded knee image.

    Returns:
        dict: Prediction label and confidence.
    """
    try:
        logger.info(f"Received knee image: {file.filename}")

        # Step 1: Read image bytes
        image_bytes = await file.read()

        # Step 2: Validate image as medical using similarity check
        similarity_response = check_similarity(image_bytes)
        similarity_result = json.loads(similarity_response.body.decode())

        logger.info(f"Similarity check result: {similarity_result}")

        if similarity_result.get("prediction") == "not-medical":
            raise HTTPException(status_code=400, detail="File is not a valid medical image.")

        # Step 3: Preprocess image
        image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_array = preprocess_image(image_pil, model_type="knee")

        # Step 4: Load knee model from app state
        knee_model = request.app.state.knee_model
        if knee_model is None:
            raise ValueError("Knee model is not loaded.")

        # Step 5: Predict
        prediction = knee_model.predict(image_array)
        predicted_index = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))

        result_label = KNEE_LABELS[predicted_index]
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
        logger.error(f"Knee prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Knee prediction failed: {str(e)}")
