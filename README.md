# Description du projet (P9 OpenClassrooms)
Développement d'un P.O.C. pour mettre en valeur un travail de veille sur le S.O.T.A. et de recherche sur les modèles de classificatio d'images en comparaison à une baseline d'un précédent projet sur un modèle fine-tuné `efficientnet-b0` sur le dataset `Flipkart`.


## Récupérer le jeu de données
Placez-vous à la racine du projet et exécutez la commande suivante pour télécharger le jeu de données dans le dossier `data`:
```bash
mkdir -p data && cd data && curl -L "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Textimage+DAS+V2/Dataset+projet+pre%CC%81traitement+textes+images.zip" -o dataset.zip
```
Le dossier `data` contient un dossier `archive` avec le jeu de données `Flipkart` téléchargé à l'URL suivante : [Dataset Flipkart Zip](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Textimage+DAS+V2/Dataset+projet+pre%CC%81traitement+textes+images.zip)

Ensuite suivez les instructions pour se placer dans le dossier `data`, dezipper le fichier, déplacer l'archive dans un dossier `archive`.<br>
A la suite de cela, le dossier Images est déplacé dans le dossier `data` ainsi le fichier `Flipkart.csv`.
```bash
cd data
unzip dataset.zip
mkdir archive
mkdir images
mv dataset.zip archive
mv Flipkart/Images/* images/
mv Flipkart/flipkart_com-ecommerce_sample_1050.csv dataset.csv
rm -rf Flipkart
```

## Environnement de travail

Afin de parfaitement isoler les dépendances propres aux expérimentations, étant donné que le projet nécessite CUDA, il m'a été obligé de séparer complètement l'environnement de travail pour les expérimentations, le développement de l'application et le développement de l'interface utilisateur.

Chacun des dossiers suivants contient un fichier `.python-version`, `pyproject.toml` et `uv.lock`:
- `backend` : Environnement de développement de l'application
- `frontend` : Environnement de développement de l'interface utilisateur
- `notebook-tf` : Environnement de développement des expérimentations sous Tensorflow
- `notebook-pytorch` : Environnement de développement des expérimentations sous Pytorch

Pour pouvoir reproduire les conditions de travail, il est nécessaire d'avoir installé `uv` en suivant les instructions suivantes:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Suite à l'installation de `uv`, il est nécessaire de redémarrer le terminal pour que la commande `uv` soit disponible.
Ensuite, il suffit de se placer dans le dossier de l'environnement de travail souhaité et d'exécuter la commande suivante:
```bash
source .venv/bin/activate
uv sync --all-groups
```

## Visualisation du résultat des expérimentations

Le suivi des expérimentations a été effectué avec `tensorboard`. Les fichiers tensorboard sont dans le dossier `runs`. Pour pouvoir visualiser les résultats, il faut soit synchroniser l'un des deux environnements de travail `notebook-tf` ou `notebook-pytorch` et exécuter la commande suivante depuis le dossier racine du projet:
```bash
tensorboard --logdir=runs
```

Il reste possible d'installer uniquement `tensorboard` depuis votre environnement de travail habituel pour consulter directement les résultats des expérimentations.

Un notebook `merged_notebook.ipynb` a été créé pour regrouper les résultats des deux expérimentations effectuées sous Tensorflow et Pytorch. Attention, ce dernier sert uniquement à visualiser les résultats. Pour tester les notebooks, il est nécessaire de se placer dans l'un des deux environnements de travail `notebook-tf` ou `notebook-pytorch` et d'intéragir avec après avoir synchronisé l'environnement de travail avec la commande suivante:
```bash
source .venv/bin/activate
uv sync --all-groups
```

## Exécution de l'application backend
Pour exécuter l'application, il est nécessaire de se placer dans le dossier `backend` et de suivre les instructions dans le README.

## Présentation de l'interface utilisateur
Pour exécuter l'interface utilisateur depuis le notebook, il est nécessaire de se placer dans le dossier `frontend` et de synchorniser l'environnement de travail avec la commande suivante:
```bash
source .venv/bin/activate
uv sync --all-groups
```

Attention, toutefois, un travail supplémentaire a été fait uniquement sur le script `main.py` et non pas le notebook qui a servi à construire les briques de l'interface utilisateur.

L'interface utilisateur a été déployée sur un service WebApp d'Azure et un workflow de CI/CD a été mis en place pour automatiser le déploiement d'un container Docker suivant le `DockerFile` et le `.dockerignore`, le push vers DockerHub et une intégration complète d'UV pour la gestion des paquets. Vous pouvez consulter le code du workflow dans le dossier `.github/workflows` et le fichier `run-build-deploy-ui.yml`.

L'ajout de la variable d'environnement `API_URL` a été fait directement depuis Azure pour pointer vers l'API FastAPI déployée. Il est nécessaire de modifier cette variable d'environnement si vous souhaitez utiliser une autre API ou si vous avez déployé l'API sur un autre service.

Les secrets GITHUB du workflow CI/CD ont été ajoutés pour permettre l'accès aux services Azure et DockerHub. Les secrets sont à paramétrer dans les paramètres du dépôt GitHub, dans la section "Secrets and variables" puis "Actions" (login/password DOCKERHUB).

L'interface utilisateur est structurée en deux parties :
- un onglet pour la classification d'images, où l'utilisateur peut charger une image parmi une liste déroulante et obtenir une prédiction de la classe de l'image avec les probabilités associées.
- un onglet pour présenter un dashboard sur le dataset `Flipkart`, où l'utilisateur peut visualiser des statistiques sur le dataset, comme le nombre d'images par classe, la taille des images, etc. Les visualisations utilisent les deux datasets sous format csv, `dataset_dashboard.csv` et `sampled.csv`, qui sont chargés dans l'interface utilisateur.

Les images de la liste déroulante sont chargées depuis le répertoire `images` et sont constituées d'un échantillon de 14 images représentatives d'au moins 2 exemplaires de chacune des 7 classes du dataset `Flipkart`.

L'interface a été codée avec `gradio` et pour les visualisations, `plotly` a été utilisé pour créer des graphiques interactifs.

## Autres dossiers
- `artifacts` : Dossier contenant les poids des modèles entraînés et les fichiers de configuration.
- `data` : Dossier contenant le jeu de données et les fichiers de configuration.
- `ressources` : Dossier contenant des éléments de recherche et sur le travail de veille accompli sur MambaVision. Le fichier `RESEARCH.md` contient un résumé des recherches effectuées et des liens vers les articles et ressources consultés.