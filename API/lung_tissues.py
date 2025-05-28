import logging
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from PIL import Image
from io import BytesIO
import numpy as np
from utils import preprocess_image
from fastapi.responses import JSONResponse

# Configure logger
logger = logging.getLogger("lung-cancer-prediction")
logger.setLevel(logging.INFO)

# Create FastAPI router
router = APIRouter()

# Define class labels
LUNG_CLASSES = {
    0: "Adenocarcinoma",
    1: "Benign",
    2: "Squamous Cell Carcinoma"
}

@router.get("/")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"message": "Lung Cancer Detection API is running"}


@router.post("/predict")
async def predict_lung_cancer(request: Request, file: UploadFile = File(...)):
    """
    Predict lung cancer type from uploaded chest CT scan image.

    Args:
        request (Request): FastAPI request to access app state.
        file (UploadFile): Uploaded image file.

    Returns:
        dict: Prediction result with label and confidence score.
    """
    try:
        logger.info(f"Received prediction request for file: {file.filename}")

        # Load the model from app state
        lung_model = request.app.state.lung_tissues_model
        if lung_model is None:
            raise ValueError("Lung cancer model is not loaded.")

        # Validate uploaded file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

        # Read image bytes and preprocess for model input
        image_bytes = await file.read()
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_array = preprocess_image(img)  # preprocess_image should prepare the image numpy array for the model

        # Perform prediction
        prediction = lung_model.predict(img_array, verbose=0)
        predicted_index = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_index])
        predicted_label = LUNG_CLASSES.get(predicted_index, "Unknown")

        logger.info(f"Prediction: {predicted_label} | Confidence: {confidence:.4f}")

        # Return prediction result
        return {
            "success": True,
            "filename": file.filename,
            "prediction": predicted_label,
            "confidence": confidence,
        }

    except HTTPException:
        # Propagate HTTP exceptions (e.g. bad request)
        raise
    except Exception as e:
        # Log and raise 500 error on unexpected exceptions
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
