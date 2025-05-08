# oc_p9
Dév. d'un P.O.C. pour mettre en valeur un travail de veille sur le S.O.T.A.


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
uv sync --all-groups
```

## Visualisation du résultat des expérimentations

Le suivi des expérimentations a été effectué avec `tensorboard`. Pour pouvoir visualiser les résultats, il faut soit synchroniser l'un des deux environnements de travail `notebook-tf` ou `notebook-pytorch` et exécuter la commande suivante depuis le dossier racine du projet:
```bash
tensorboard --logdir=runs
```

Il reste possible d'installer uniquement `tensorboard` depuis votre environnement de travail habituel pour consulter directement les résultats des expérimentations.
