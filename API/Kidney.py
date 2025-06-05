from fastapi import FastAPI, File, UploadFile, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import json
import logging
from PIL import Image
from io import BytesIO


from utils import preprocess_image  # Reuse preprocessing from utils
from similarity import check_similarity  # Similarity checker function

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a router for kidney-related prediction
router = APIRouter()

# Labels corresponding to kidney model output
labels_map = {
    0: "could be Cyst",
    1: "No abnormal findings detected",
    2: "could be Stone",
    3: "could be Tumor"
}


@router.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Kidney Detection API is running"}

@router.post("/predict")
async def predict_kidney(request: Request, file: UploadFile = File(...)):
    """
    Endpoint to predict kidney scan class:
    - First checks if the image is a medical scan.
    - If valid, predicts its class using the kidney model.
    """
    try:
        logger.info(f"Received file: {file.filename}")

        # Read image content
        contents = await file.read()

        # Step 1: Run medical image similarity check
        similarity_response = check_similarity(contents)
        similarity_data = similarity_response.body.decode()
        similarity_result = json.loads(similarity_data)

        # Reject non-medical images
        if similarity_result.get("prediction") == "not-medical":
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not recognized as a valid medical image."
            )

        logger.info(f"Similarity check passed: {similarity_result}")

        # Step 2: Load and preprocess image
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_array = preprocess_image(img,model_type="resnet") # Use ResNet preprocessing as per model requirements

        # Step 3: Retrieve kidney model from FastAPI app state
        kidney_model = request.app.state.kidney_model
        if kidney_model is None:
            raise ValueError("Kidney model is not loaded in app state.")

        # Step 4: Run prediction
        prediction = kidney_model.predict(img_array)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        # Step 5: Return result
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
