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

import warnings
import base64
from io import BytesIO
from fastapi import FastAPI
import torch
from classes import (
    MambaClassifier,
    PayloadRequest,
    NormalizationResponse,
    PredictionResponse,
)
from transformers import AutoModel
from constants import MAMBA_HIDDEN_SIZES, MODEL_CARD, MODEL_PATH
from constants import N_CLASSES, LABEL2IDX, LABELS
from PIL import Image

# FastAPI application instance
app = FastAPI(title="Image Prediction API")

# Ignore FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Assign the device based on availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to decode base64 string to image
def decode_base64_to_image(base64_str: str) -> Image.Image:
    """Decode a base64-encoded string to a PIL Image."""
    image_bytes = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_bytes))


# Function to convert image to base64 string
def img_to_base64(img):
    """Convert a PIL Image to a base64-encoded JPEG string."""
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Transform the tensor to PIL image
def tensor_to_pil(tensor):
    """Convert a torch tensor to a PIL Image."""
    tensor = tensor.squeeze(0)
    tensor = tensor.permute(1, 2, 0)  # Change the order of dimensions
    tensor = (tensor * 255).byte().numpy()  # Convert to uint8
    return Image.fromarray(tensor)


# Loading the model on startup
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


@app.get("/")
async def root():
    """
    Root endpoint for the API.

    Returns:
        dict: A simple message indicating that the API is running.
    """
    return {"message": "API en cours d'exécution"}

@app.post("/normalize", response_model=NormalizationResponse)
async def normalize(
    payload: PayloadRequest,
):
    """
    Asynchronously processes an uploaded image.
    """
    # Image preprocessing
    img = decode_base64_to_image(payload.image)
    image_normalized = app.state.transform(img).unsqueeze(0)

    # Apply tensor to PIL Image to Base64
    image_normalized = tensor_to_pil(image_normalized)
    image_normalized = img_to_base64(image_normalized)

    return NormalizationResponse(image_normalized=image_normalized)

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    payload: PayloadRequest,
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
    img = decode_base64_to_image(payload.image)
    image_normalized = app.state.transform(img).unsqueeze(0).to(DEVICE)

    # Perform inference
    with torch.no_grad():
        outputs = app.state.model(image_normalized)
        _, predicted = torch.max(outputs, 1)
        pred_idx = predicted.item()
        predicted_label = LABELS[pred_idx]
        probs_tensor = torch.nn.functional.softmax(outputs, dim=1)
        probs_serialized = probs_tensor.squeeze(0).cpu().numpy().tolist()
        probabilities = {label: round(probs_serialized[idx], 2) for label, idx in LABEL2IDX.items()}

    return PredictionResponse(predicted_label=predicted_label, probabilities=probabilities)
