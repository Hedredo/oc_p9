"""
Image Prediction API

This FastAPI application provides an endpoint for predicting the label of an uploaded image
using a pre-trained PyTorch model. The application includes the following features:

- **Startup Event**: Loads the pre-trained model into memory when the application starts.
- **Prediction Endpoint**: Accepts an image file and an expected label, processes the image,
    and returns the predicted label along with the received label.

Modules:
- `FastAPI`: Framework for building the API.
- `torch`: Used for loading the model and performing inference.
- `PIL.Image`: For image processing.
- `pydantic.BaseModel`: For defining the response model.

Endpoints:
- **POST /predict**:
        - **Description**: Predicts the label of an uploaded image.
        - **Request Parameters**:
                - `image` (UploadFile): The image file in JPEG or PNG format.
                - `label` (str): The expected label for the image.
        - **Response**:
                - `received_label` (str): The label provided in the request.
                - `predicted_label` (str): The label predicted by the model.
        - **Error Handling**:
                - Returns a 400 status code if the uploaded file is not a valid image.

Startup:
- The `load_model` function is triggered on application startup to load the pre-trained
    PyTorch model (`model.pth`) and set it to evaluation mode.

Usage:
1. Start the FastAPI application.
2. Use the `/predict` endpoint to upload an image and receive a prediction.
"""

import io
import warnings
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import torch
from classes import MambaClassifier, PredictionResponse
from transformers import AutoModel
from constants import MAMBA_HIDDEN_SIZES, MODEL_CARD, MODEL_PATH
from constants import N_CLASSES, LABEL2IDX, LABELS
from PIL import Image

app = FastAPI(title="Image Prediction API")

# Suppression des avertissements
warnings.filterwarnings("ignore", category=FutureWarning)
# Déclaration globale du device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Chargement du modèle au démarrage
@app.on_event("startup")
async def load_model():
    """
    Handles the startup event to initialize and load the machine learning model.

    This function performs the following steps:
    1. Initializes the `MambaClassifier` model with the specified pre-trained model,
        number of classes, and hidden dimensions.
    2. Creates a transformation function for preprocessing input data.
    3. Loads the model weights from the specified file path.
    4. Sets the model to evaluation mode and moves it to the appropriate device.

    Attributes:
        app.state.model (MambaClassifier): The initialized and loaded model.
        app.state.transform (callable): The transformation function for preprocessing input data.

    Raises:
        FileNotFoundError: If the model weights file specified by `MODEL_PATH` is not found.
        RuntimeError: If there is an issue loading the model weights or moving the model to the device.
    """
    # Define the model with the MambaClassifier class
    app.state.model = MambaClassifier(
        AutoModel.from_pretrained(MODEL_CARD, trust_remote_code=True),
        num_classes=N_CLASSES,
        hidden_dim=MAMBA_HIDDEN_SIZES.get(MODEL_CARD),
    )
    # Create the transform function
    app.state.transform = app.state.model.create_transform(
        training=False,
        auto_augment=None,
    )
    # Load the model weights
    app.state.model.load_state_dict(
                    torch.load(MODEL_PATH),
                )
    # Set the model to evaluation mode and move it to the appropriate device
    app.state.model.eval()
    app.state.model.to(DEVICE)

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(..., description="Image au format JPEG ou PNG"),
    label: str = Form(..., description="Label attendu pour l'image")
):
    """
    Asynchronously processes an uploaded image, performs inference using a pre-trained model, 
    and returns the predicted label along with the received label.
    Args:
        image (UploadFile): The uploaded image file in JPEG or PNG format.
        label (str): The expected label for the uploaded image.
    Returns:
        PredictionResponse: A response object containing the received label and the predicted label.
    Raises:
        HTTPException: If the uploaded image is invalid or cannot be processed.
    """
    # Lecture et conversion de l’image
    contents = await image.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Image non valide")

    # Image preprocessing
    img_processed = app.state.transform(img).unsqueeze(0).to(DEVICE)

    # Perform inference
    with torch.no_grad():
        outputs = app.state.model(img_processed)
        _, predicted = torch.max(outputs, 1)
        pred_idx = predicted.item()
        predicted_label = LABELS[pred_idx]

    return PredictionResponse(
        received_label=label,
        predicted_label=predicted_label
    )
