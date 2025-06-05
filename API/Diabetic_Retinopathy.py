import logging
from fastapi import APIRouter, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import numpy as np
from utils import preprocess_image  

# Setup logger
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()

# Class labels for prediction
DR_CLASSES = {
    0: "could be Diabetic Retinopathy",
    1: "Healthy"
}

@router.get("/")
async def root():
    """Health check route"""
    return {"message": "Diabetic Retinopathy Detection API is running"}

@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Predict if retinal image shows signs of diabetic retinopathy.
    """
    try:
        logger.info(f"Received file: {file.filename}")

        # Ensure the uploaded file is an image
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

        # Read image content
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")

        # Preprocess the image using shared utils
        img_array = preprocess_image(img,model_type="resnet")

        # Access the loaded model from app state
        dr_model = request.app.state.Diabetic_Retinopathy_model
        if dr_model is None:
            raise ValueError("Diabetic Retinopathy model not loaded in app state")

        # Make prediction
        prediction = dr_model.predict(img_array, verbose=0)
        predicted_class = int(prediction[0][0] > 0.5)
        confidence = float(prediction[0][0])

        result = DR_CLASSES[predicted_class]
        logger.info(f"Prediction result: {result} (confidence: {confidence:.4f})")

        return {
            "success": True,
            "filename": file.filename,
            "prediction": result,
            "confidence": confidence
        }

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
