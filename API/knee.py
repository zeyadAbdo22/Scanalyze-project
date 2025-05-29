from fastapi import FastAPI, File, UploadFile, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import json
import logging
from PIL import Image
from io import BytesIO

from utils import preprocess_image  # Now MobileNet-compatible
from similarity import check_similarity  # Image type checker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Knee model output labels
labels_map = {
    0: "Healthy",
    1: "could have Osteoporosis"
}

@router.get("/")
async def root():
    """Health check route"""
    return {"message": "Knee Detection API is running"}

@router.post("/predict")
async def predict_knee(request: Request, file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        contents = await file.read()

        # Check if image is medical
        similarity_response = check_similarity(contents)
        similarity_data = similarity_response.body.decode()
        similarity_result = json.loads(similarity_data)

        if similarity_result.get("prediction") == "not-medical":
            raise HTTPException(status_code=400, detail="Not a valid medical image")

        logger.info(f"Similarity result: {similarity_result}")

        # Preprocess image using shared utils
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_array = preprocess_image(img, model_type="mobilenet")  # MobileNet-compatible 

        # Get model from app state
        knee_model = request.app.state.knee_model
        if knee_model is None:
            raise ValueError("Knee model not loaded")

        # Predict class
        prediction = knee_model.predict(img_array)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return {
            "success": True,
            "filename": file.filename,
            "prediction": labels_map[predicted_class],
            "confidence": confidence
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
