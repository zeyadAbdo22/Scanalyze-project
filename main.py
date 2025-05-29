import logging
import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from utils import load_model_from_azure  # Custom function to load models from Azure

# Import API routers
from API.brain_xray import router as brain_router
from API.lung_xray import router as lung_xray_router
from API.lung_tissues import router as lung_tissues_router
from API.Kidney import router as kidney_router
from API.knee import router as Knee_router
from API.Diabetic_Retinopathy import router as Diabetic_Retinopathy_router

# ------------------------ ENVIRONMENT SETUP ------------------------

# Force TensorFlow to run on CPU and avoid oneDNN issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Setup logging format and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Reduce verbosity from TensorFlow logs
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# ------------------------ FASTAPI APP SETUP ------------------------

app = FastAPI(title="Medical Analysis API")

# Enable CORS for all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------ MODEL LOADING ON STARTUP ------------------------

@app.on_event("startup")
async def startup_event():
    """Load all models from Azure Blob Storage on app startup."""
    try:
        logging.info("Initializing model loading...")

        app.state.brain_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/resnet_brain_model.h5"
        )
        logging.info("Brain Tumor model loaded successfully")

        app.state.lung_xray_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/resnet50_lung_xray.h5"
        )
        logging.info("Lung X-Ray model loaded successfully")

        app.state.lung_tissues_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/lung-cancer-resnet-model.h5"
        )
        logging.info("Lung Tissues model loaded successfully")

        app.state.kidney_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/resnet50_kidney_ct_augmented.h5"
        )
        logging.info("Kidney model loaded successfully")

        app.state.knee_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/Knee_Osteoporosis.h5"
        )
        logging.info("Knee model loaded successfully")

        app.state.Diabetic_Retinopathy_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/Diabetic-Retinopathy-ResNet50-model.h5"
        )
        logging.info("Diabetic Retinopathy model loaded successfully")

    except Exception as e:
        logging.error(f"Error during model loading: {str(e)}")
        raise

# ------------------------ ROOT ENDPOINT ------------------------

@app.get("/")
async def root():
    """Root route for health check"""
    return {
        "status": "healthy",
        "message": "Medical Scan Detection API is running",
    }

# ------------------------ API ROUTES ------------------------

# Mount all routers for different prediction endpoints
app.include_router(brain_router, prefix="/brain-XRays", tags=["Brain Detection"])
app.include_router(lung_xray_router, prefix="/Lung-XRays", tags=["Lung X-Ray Detection"])
app.include_router(lung_tissues_router, prefix="/Lung-tissues", tags=["Lung Tissues Detection"])
app.include_router(kidney_router, prefix="/kidney", tags=["Kidney Detection"])
app.include_router(Knee_router, prefix="/Knee", tags=["Knee Detection"])
app.include_router(Diabetic_Retinopathy_router, prefix="/Diabetic-Retinopathy", tags=["Diabetic Retinopathy Detection"])
