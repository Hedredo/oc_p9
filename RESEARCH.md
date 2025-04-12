# Introduction
Ce fichier markdown contient un résumé des recherches effectuées pour le projet MambaVision. Il est divisé en plusieurs sections, chacune correspondant à source différente.<br>
Chaque section pourra contenir des images, copies d'écran, liens et autres ressources pertinentes pour le projet.<br>
Il sera établie une synthèse de chaque source qui pourra alimenter un travail ultérieur sur le projet en les utilisant comme contexte lors d'un RAG.<br>

# Lexique
## Backbone / Encoder :
Architecture principale qui extrait des représentations compactes et informatives d'une image. Les premiers backbones étaient basés sur des réseaux de neurones convolutifs (CNN), mais les architectures modernes utilisent souvent des transformateurs, qui sont plus efficaces pour capturer les relations à long terme dans les données visuelles.
Liste des backbones :
- Deep Convolutional Neural Networks (DCNN) : VGG, ResNet, Inception, EfficientNet, etc.
- Vision Transformers (ViT) : ViT, DeiT, Swin Transformer, etc.
- Vision Language Models (VLM) : CLIP, ALIGN, etc.
- Vision Language Transformer (ViLT)
- Mamba ? ```TODO```
- SSM ? ```TODO```
- Hybrid Mamba Transformer ? ```TODO```

## Decoder :
Architecture qui prend les représentations extraites par le backbone et les reconstruit en une sortie de la même taille que l'entrée. Dans le cas de la segmentation sémantique, le décodeur reconstruit une image à partir des caractéristiques extraites par le backbone, permettant ainsi de prédire des masques de segmentation pour chaque pixel de l'image d'entrée.

## Bottleneck :
Dans U-Net, on aborde souvent le concept de "bottleneck" qui est une couche intermédiaire entre l'encodeur et le décodeur. Il s'agit d'une couche qui réduit la dimensionnalité des données tout en préservant les informations essentielles. 

## MLP head :
Le Multi Layer Perceptron (MLP) est un type de réseau de neurones qui se compose de plusieurs couches Fully-Connected. Il est souvent utilisé comme tête de classification dans les architectures de réseaux de neurones pour effectuer des tâches telles que la classification d'images ou la régression. On parle de classification head, segmentation head, etc. selon la tâche à accomplir.

## Transformer :
Architecture de réseau de neurones qui utilise des mécanismes d'attention pour traiter les données séquentielles. Les transformateurs sont devenus populaires dans le traitement du langage naturel et sont maintenant utilisés dans la vision par ordinateur.

## Mamba :
```TODO```

## SSM - Selective State Model
```TODO```


# Ressources
=================
## 1. Papers & Publications
### 1.1. NVIDIA Research - MambaVision: A Hybrid Mamba Transformer Vision Backbone
URL : https://research.nvidia.com/publication/2025-06_mambavision-hybrid-mamba-transformer-vision-backbone

![architecture](ressources/ManbaVisionArchitecture.png)

Synthèse:
- MambaVision est un backbone hybride qui combine les avantages des Mamba et des transformateurs pour la vision par ordinateur.
- Il est conçu pour être efficace en termes de calcul et de mémoire, tout en offrant des performances de pointe sur plusieurs tâches de vision.
- Démonstration via une étude d'ablation des composants de l'architecture et prouver l'efficacité d'intégérer d'intégrer dans les blocks MAMBA des blocks de self-attention propres aux ViT améliore ce qui faisait défaut à MAMBA.
- MambaVision sur ImageNet 1K SOTA top-1 accuracy (87.4% et 2.5x plus rapide que le ViT de base ????). ```TODO```
- Performe de la classification d'images à la segmentation sémantique, en passant par la détection d'objets et la segmentation d'instance.

Auteurs :
- Ali Hatamizadeh
- Jan Kautz

Contexte des princpipaux de papiers qui ont précédé :
- ![Convolution S.S.M.](https://research.nvidia.com/publication/2023-12_convolutional-state-space-models-long-range-spatiotemporal-modeling)
- ![Mamba-based Language Models](https://research.nvidia.com/publication/2024-06_empirical-study-mamba-based-language-models)
- ![Gated delta Networks with Mamba](https://research.nvidia.com/publication/2025-04_gated-delta-networks-improving-mamba2-delta-rule)


## 2. Courses & Tutorials
### 2.1. Analytics Vidhya - Building LLM with Mamba & SSM
URL : https://www.analyticsvidhya.com/blog/2024/01/mamba-a-selective-state-space-model-for-vision-transformers/

Attention, ce cours se concentre sur les LLM mais il est une bonne introduction à l'architecture Mamba et à son fonctionnement. Il aborde les concepts de base des Mamba et des SSM, ainsi que leur application dans le contexte des LLM.


Synthèse Introduction:
- Désavantage de l'architecture Transformers : la complexité quadratique de l'attention. A chaque étape de l'attention, le modèle doit calculer les relations entre le nouveau token généré et tous les tokens précédents, ce qui peut devenir coûteux en termes de calcul et de mémoire. La complexité quadratique de l'attention rend difficile le traitement de longues séquences de données.
- La structure linéaire du RNN : Les RNN traitent les données séquentiellement, en mettant à jour l'état caché à chaque étape de la séquence et génèrent un nouvel état caché en fonction de l'état précédent et du nouvel élément d'entrée. Cela permet aux RNN de capturer les relations à long terme dans les données séquentielles. Le désavantage est que l'information est compréhensée dans l'hidden state au fur est à mesure que la séquence avance, ce qui peut entraîner une perte d'information et de précision à terme.

Synthse State Space Model:
- Un State Space est un modèle mathématique qui décrit la dynamique d'un système en termes d'états internes et d'entrées/sorties. Un State Vector est un vecteur qui représente l'état interne du système à un moment donné qui comprends les coordonnées X,Y de l'objet dans l'espace et aussi la distance nécessaire juqu'à la sortie.

- Reprendre l'idée de l'architecture RNN (Recurrent Neural Network) qui traite les données séquentiellement, en se concentrant sur les relations locales entre les pixels plutôt que sur l'ensemble de l'image.
- L'architecture Mamba utilise des blocs de convolution pour extraire des caractéristiques locales, suivis d'une attention globale pour capturer les relations à long terme.