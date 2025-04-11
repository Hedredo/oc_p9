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

## Prétraitement

