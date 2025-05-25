"""
Frontend application for image classification and dashboard visualization using Gradio.
Provides UI for image selection, classification, transformation, and dataset exploration.
"""

import os
from pathlib import Path
import base64
import json
from io import BytesIO

import requests
import gradio as gr
from PIL import Image
import pandas as pd
import plotly.express as px

from constants import CLOUD_FOLDER, API_URL

# Load the dataset for the dashboard
df_dashboard = pd.read_csv(CLOUD_FOLDER / "dataset_dashboard.csv")

FONT_FAMILY = "Segoe UI, Arial, Verdana, Helvetica, sans-serif"

# Draw a scatter plot with height and width from df_dashboard as x and y axis
fig = px.scatter(
    df_dashboard,
    x="largeur",
    y="hauteur",
    color="Catégorie",
    hover_name="Nom du produit",
    hover_data=["fichier"],
    color_discrete_sequence=px.colors.qualitative.Safe,  # Colorblind-friendly palette
)
fig.update_traces(marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")))
fig.update_layout(
    title=dict(
        text="Résolution des images",
        font=dict(size=16, color="black", family=FONT_FAMILY, weight="bold"),
        xanchor="center",
        x=0.4,
    ),
    width=850,
    height=500,
    xaxis_title="Largeur (pixels)",
    yaxis_title="Hauteur (pixels)",
    legend_title=dict(
        text="Catégories<br>(cliquez pour filtrer)",
        font=dict(size=13, color="black", family=FONT_FAMILY, weight="bold"),
    ),
    font=dict(
        family=FONT_FAMILY,
        size=12,
        color="black",
    ),
)

# draw a bar plot with the number of images per class (all bars same height, no image names)
df_bar = (
    df_dashboard.groupby("Catégorie", observed=False)
    .size()
    .reset_index(name="fréquence")
)
fig2 = px.bar(
    df_bar,
    x="Catégorie",
    y="fréquence",
    color="Catégorie",
    color_discrete_sequence=px.colors.qualitative.Safe,
)
fig2.update_layout(
    title=dict(
        text="Nombre d'images par catégorie",
        font=dict(size=16, color="black", family=FONT_FAMILY, weight="bold"),
        xanchor="center",
        x=0.4,
    ),
    width=850,
    height=500,
    xaxis_title="Catégorie",
    yaxis_title="Nombre d'images",
    legend_title=dict(
        text="Catégories<br>(cliquez pour filtrer)",
        font=dict(size=13, color="black", family=FONT_FAMILY, weight="bold"),
    ),
    font=dict(
        family=FONT_FAMILY,
        size=12,
        color="black",
    ),
)

# Load the sampled dataset and create mappings for image paths and IDs
df = pd.read_csv("sampled.csv")
files = os.listdir(CLOUD_FOLDER / "images")
filename2path = {filename: CLOUD_FOLDER / "images" / filename for filename in files}
id2filename = {
    df.loc[df["image"] == filename, "image_renamed"].values[0]: filename
    for filename in files
}


# Define the function used in the Gradio interface
def load_image(name: str) -> Path:
    """
    Load an image from the specified path.
    """
    filename = id2filename.get(name)
    filepath = filename2path.get(filename)
    return Image.open(filepath)


# Function to decode base64 string to image
def decode_base64_to_image(base64_str: str) -> Image.Image:
    """Decode a base64-encoded string to a PIL Image."""
    image_bytes = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_bytes))


# Function to convert image to base64 string
def img_to_base64(img: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded JPEG string."""
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# Create a request with the img_b64 as a payload to the API
def send_image_to_api(image_b64: str, endpoint: str, api_url: str = API_URL) -> str:
    """Send a base64 image to the API and return the JSON response."""
    url = f"{api_url}/{endpoint}/"
    headers = {"Content-Type": "application/json"}
    payload = json.dumps({"image": image_b64})
    response = requests.post(url, headers=headers, data=payload, timeout=120)
    return response.json()


def classify_image(image_name: str) -> str:
    """
    Predict the label of the image using a dummy model.
    """
    # Simulate a prediction
    img = load_image(image_name)
    img_b64 = img_to_base64(img)
    # Send the image to the API
    content = send_image_to_api(img_b64, "predict")
    # Extract the label from the response
    label = content.get("predicted_label", "Unknown")
    probs = content.get("probabilities", "Unknown")
    df_probs: pd.DataFrame = pd.DataFrame(
        {"Catégorie": probs.keys(), "Probabilité": probs.values()}
    )
    return label, df_probs


def transform_image(image_name: str) -> str:
    """
    Transform the image using a dummy model.
    """
    img = load_image(image_name)
    img_b64 = img_to_base64(img)
    content = send_image_to_api(img_b64, "normalize")
    # Extract the transformed image from the response
    transformed_img_b64 = content.get("image_normalized", "Unknown")
    # Decode the base64 string to an image
    # Convert the image to PIL
    transformed_img = Image.open(BytesIO(base64.b64decode(transformed_img_b64)))
    return transformed_img

    # Gradio code pour l'interface utilisateur


with gr.Blocks(
    css="#image_input {border: 2px solid #ccc;}",
    js="""
    () => {
        setTimeout(() => {
            // Tooltip pour l'image
            const img = document.querySelector('#image_input img');
            if (img) {
                img.title = "image à classifier";
            }
            // Tooltip pour bloc avec la catégorie prédite
            const label = document.querySelector('#image_label');
            if (label) {
                label.title = "Catégorie prédite";
            }
            // Tooltip pour le bouton "Classifier l'image"
            const btns = document.querySelectorAll('button');
            btns.forEach(btn => {
                if (btn.innerText.trim() === "Classifier l'image") {
                    btn.setAttribute('title', "Classifier l'image");
                }
            });
        }, 500);
    }
    """,
) as demo:
    # Application title. TODO : Ajouter la description de l'app
    gr.Markdown(
        """
        <h1 style="text-align: center; color: #4CAF50;">Mon App</h1>
        """,
        elem_id="title",  # ID pour le CSS
        elem_classes="title",  # Classes CSS pour le style
    )
    #####################
    # Classifier tab  #
    #####################
    with gr.Tab("Classification"):
        gr.Markdown(
            """
            <h2 style="text-align: left;">Interface applicative pour utiliser le modèle de classification d'image.</h2>
            <p style="text-align: left;">Instructions ...</p>
            """,
            elem_id="classifier_title",  # ID pour le CSS
            elem_classes="classifier-title",  # Classes CSS pour le style
        )
        # List of images to classify
        with gr.Row():
            dropdown = gr.Dropdown(
                label="Choisissez un produit dans la liste déroulante à afficher",
                choices=list(id2filename.keys()),  # Liste des labels
                value=list(id2filename.keys())[0],  # Valeur par défaut
            )
        # Block to display the image and the predicted label
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    value=list(filename2path.values())[0],
                    type="filepath",
                    label="Image à classifier",
                    elem_id="image_input",  # ID pour le CSS
                    elem_classes="image-input",  # Classes CSS pour le style
                    height=300,
                    width=300,
                )
            with gr.Column():
                with gr.Row():
                    image_label = gr.Label(
                        label="Catégorie prédite",
                        elem_id="image_label",  # ID pour le CSS
                        elem_classes="image-label",  # Classes CSS pour le style
                    )
                with gr.Row():
                    # Display the predicted probabilities for each class with a bar chart
                    df_output = gr.BarPlot(
                        label="Scores par classe",
                        y="Catégorie",
                        x="Probabilité",
                    )

            # Charger l'exemple sélectionné
            dropdown.change(fn=load_image, inputs=dropdown, outputs=image_input)
        # Button to launch the classification
        with gr.Row():
            # Bouton pour lancer la segmentation
            submit_btn = gr.Button("Classifier l'image")
            submit_btn.click(
                classify_image, inputs=dropdown, outputs=[image_label, df_output]
            )

    #################
    # Dashboard tab #
    #################
    with gr.Tab("Dashboard"):
        gr.Markdown(
            """
            <h2 style="text-align: left;">Dashboard de présentation du dataset Flipkart.</h2>
            <p style="text-align: left;">
            Cette section présente des visualisations interactives du dataset Flipkart, permettant d'explorer la distribution des images et des catégories.
            </p>
            """,
            elem_id="dashboard_title",  # ID pour le CSS
            elem_classes="dashboard-title",  # Classes CSS pour le style
        )
        with gr.Row():
            gr.Markdown(
                """
                <h3 style="text-align: left;">Résolution des images</h3>
                <p style="text-align: left;">
                Ce graphique montre la répartition des résolutions (largeur et hauteur) des images du dataset, colorées par catégorie.
                </p>
                """,
                elem_id="scatterplot_title",  # ID pour le CSS
                elem_classes="scatterplot-title",  # Classes CSS pour le style
            )
        with gr.Row():
            gr.Plot(
                value=fig,
                label="Scatterplot des tailles d'images",
                elem_id="scatterplot_dashboard",
                elem_classes="scatterplot-dashboard",  # Classes CSS pour le style
            )
        with gr.Row():
            gr.Markdown(
                """
                <h3 style="text-align: left;">Nombre d'images par catégorie</h3>
                <p style="text-align: left;">
                Ce graphique à barres affiche le nombre d'images disponibles pour chaque catégorie dans le dataset.
                </p>
                """,
                elem_id="barplot_title",  # ID pour le CSS
                elem_classes="barplot-title",  # Classes CSS pour le style
            )
        with gr.Row():
            gr.Plot(
                value=fig2,
                label="Histogramme du nombre d'images par catégorie",
                elem_id="barplot_dashboard",
                elem_classes="barplot-dashboard",  # Classes CSS pour le style
            )
        # Block to display the image and the transformed image
        with gr.Row():
            gr.Markdown(
                """
                <h3 style="text-align: left;">Image transformée</h3>
                <p style="text-align: left;">Reproduction des prétraitements effectués par le modèle MambaVision</p>
                """,
                elem_id="image_transformed_title",  # ID pour le CSS
                elem_classes="image_transformed_title",  # Classes CSS pour le style
            )
        # List of images to classify
        with gr.Row():
            dropdown = gr.Dropdown(
                label="Choisissez un produit dans la liste déroulante à afficher",
                choices=list(id2filename.keys()),  # Liste des labels
                value=list(id2filename.keys())[0],  # Valeur par défaut
            )
        with gr.Row():
            image_input = gr.Image(
                value=list(filename2path.values())[0],
                type="filepath",
                label="Image à classifier",
                elem_id="image_input",  # ID pour le CSS
                elem_classes="image-input",  # Classes CSS pour le style
                height=300,
                width=300,
            )
            image_transformed = gr.Image(
                type="pil",
                label="Image transformée",
                elem_id="image_input",  # ID pour le CSS
                elem_classes="image-input",  # Classes CSS pour le style
                height=300,
                width=300,
            )
            # Charger l'exemple sélectionné
            dropdown.change(fn=load_image, inputs=dropdown, outputs=image_input)
        # Button to launch the classification
        with gr.Row():
            # Bouton pour lancer la segmentation
            submit_btn = gr.Button("Transformer l'image")
            submit_btn.click(
                transform_image, inputs=dropdown, outputs=[image_transformed]
            )


if __name__ == "__main__":
    demo.launch(
        share=False,
        allowed_paths=[CLOUD_FOLDER],
        server_name="0.0.0.0",
        server_port=7860,
    )
