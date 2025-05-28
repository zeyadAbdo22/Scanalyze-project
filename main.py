from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

from utils import load_model_from_azure

# Routers for different medical scan endpoints
from API.brain_xray import router as brain_router
from API.lung_xray import router as lung_xray_router
from API.lung_tissues import router as lung_tissues_router
from API.Kidney import router as kidney_router
from API.knee import router as knee_router
from API.Diabetic_Retinopathy import router as retina_router

# Disable GPU and ONEDNN optimizations
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Create FastAPI app instance
app = FastAPI(title="Medical Scan Detection API")

# Allow all CORS origins (change in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def load_models():
    """Load all models from Azure on application startup."""
    try:
        logging.info("Loading models from Azure...")

        app.state.brain_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/resnet_brain_model.h5"
        )
        app.state.lung_xray_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/resnet50_lung_xray.h5"
        )
        app.state.lung_tissues_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/lung-cancer-resnet-model.h5"
        )
        app.state.kidney_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/resnet50_kidney_ct_augmented.h5"
        )
        app.state.knee_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/Knee_Osteoporosis.h5"
        )
        app.state.retina_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/Diabetic-Retinopathy-ResNet50-model.h5"
        )

        logging.info("All models loaded successfully.")

    except Exception as e:
        logging.error(f"Failed to load models: {str(e)}")
        raise

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Medical Scan Detection API is running"
    }

# Register routers with prefixes
app.include_router(brain_router, prefix="/brain-XRays", tags=["Brain Detection"])
app.include_router(lung_xray_router, prefix="/Lung-XRays", tags=["Lung XRays Detection"])
app.include_router(lung_tissues_router, prefix="/Lung-tissues", tags=["Lung Tissues Detection"])
app.include_router(kidney_router, prefix="/kidney", tags=["Kidney Detection"])
app.include_router(knee_router, prefix="/Knee", tags=["Knee Detection"])
app.include_router(retina_router, prefix="/Diabetic-Retinopathy", tags=["Diabetic Retinopathy Detection"])
