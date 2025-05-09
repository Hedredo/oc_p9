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

Note:
- Ensure the `model.pth` file is available in the working directory.
- The `preprocess` function should be defined elsewhere to handle image preprocessing.
"""
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import torch
from classes import MambaClassifier
from transformers import AutoModel
from constants import MAMBA_HIDDEN_SIZES, MODEL_CARD, MODEL_PATH
from constants import N_CLASSES, LABEL2IDX, LABELS
from PIL import Image

app = FastAPI(title="Image Prediction API")

# Déclaration globale du device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement du modèle au démarrage
@app.on_event("startup")
async def load_model():
    global model, preprocess

    # Define the model with the MambaClassifier class
    model = MambaClassifier(
        AutoModel.from_pretrained(MODEL_CARD, trust_remote_code=True),
        num_classes=N_CLASSES,
        hidden_dim=MAMBA_HIDDEN_SIZES.get(MODEL_CARD),
    )

    model.load_state_dict(
                    torch.load(MODEL_PATH),
                )
    model.eval()
    model.to(DEVICE)

class PredictionResponse(BaseModel):
    received_label: str
    predicted_label: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(..., description="Image au format JPEG ou PNG"),
    label: str = Form(..., description="Label attendu pour l'image")
):
    # Lecture et conversion de l’image
    contents = await image.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Image non valide")

    # Define the configuration for the model
    processor = model.create_transform(training=False, auto_augment=None)
    img_processed = processor(img).unsqueeze(0).to(DEVICE)
    # Perform inference
    with torch.no_grad():
        outputs = model(img_processed)
        _, predicted = torch.max(outputs, 1)
        pred_idx = predicted.item()
        predicted_label = LABELS[pred_idx]

    return PredictionResponse(
        received_label=label,
        predicted_label=predicted_label
    )
