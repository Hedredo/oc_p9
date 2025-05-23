**Servir l'API FastAPI avec ngrok (1) ou serveo (2)**
========================================

# 1. NGROK

Il est nécessaire d'avoir un compte sur le site de ngrok pour pouvoir utiliser ngrok. Il est également nécessaire d'avoir installé ngrok sur votre machine.
Le token d'authentification est nécessaire pour pouvoir utiliser ngrok.
Ces étapes sont à exécuter sur la machine où le serveur API est lancé.

## 1.1. Donwload and Install
```bash
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
  | sudo tee /etc/apt/sources.list.d/ngrok.list \
  && sudo apt update \
  && sudo apt install ngrok
```

## 1.2. Authentificate

Il est nécessaire de s'authentifier pour utiliser ngrok. Il faut donc créer un compte sur le site de ngrok et récupérer le token d'authentification.<br>
Par la suite il est nécessaire d'éxécuter la commande suivante pour l'authentification :
```bash
ngrok config add-authtoken <token>
```

## 1.3. Start ngrok
Il est nécessaire de lancer le serveur API avant de lancer ngrok. Il est nécessaire d'adapter le port en fonction du port sur lequel le serveur API est lancé.Par défaut, le serveur API est lancé sur le port 8000.

```bash
fastapi run main.py
```


Pour obtenir une adresse publique aléatoire et temporaire qui redirige vers le serveur API local, il est nécessaire de lancer la commande suivante :
```bash
ngrok http http://localhost:8000
```

Une adresse publique fixe a été créé avec ngrok pour le serveur API. Pour l'utiliser, il est nécessaire de lancer la commande suivante :
```bash
ngrok http --url=valid-flowing-mantis.ngrok-free.app 80
```


# 2. SERVEO
SERVEO ne nécessite pas de compte ni d'authentification. Il est donc possible de l'utiliser directement. Il est également nécessaire d'avoir installé ssh sur votre machine.
Il est nécessaire de lancer le serveur API avant de lancer serveo. Il est nécessaire d'adapter le port en fonction du port sur lequel le serveur API est lancé. Par défaut, le serveur API est lancé sur le port 8000.

```bash
ssh -R 80:localhost:8000 serveo.net
```
Ensuite serveo renvoie une adresse publique qui redirige vers le serveur API local. Il est donc possible d'utiliser cette adresse pour accéder au serveur API depuis l'extérieur.

Ex. : https://4aa9c87fca515c0ac5721662850b4a91.serveo.net/docs

# Lancer le serveur API
Pour lancer le serveur API, il est nécessaire d'exécuter la commande suivante :
```bash
fastapi run main.py
```
# Documentation de l'API
La documentation de l'API est accessible à l'adresse suivante : 
```
http://localhost:8000/docs
```
# Accéder à l'API depuis l'extérieur
Pour accéder à l'API depuis l'extérieur, il est nécessaire d'utiliser l'adresse publique fournie par ngrok ou serveo. Par exemple, si vous utilisez ngrok, l'adresse publique sera de la forme :
```
https://valid-flowing-mantis.ngrok-free.app/docs
```