# Utiliser une image Python 3.11 slim comme base
FROM python:3.11-slim-bullseye

# Mettre à jour les paquets système pour corriger les vulnérabilités
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Installer uv (binaire officiel)
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de gestion des dépendances
COPY .python-version pyproject.toml uv.lock ./

# Installer les dépendances avec uv
RUN uv sync --locked

# Copier le reste de l'application
COPY . .

# Définir la variable d'environnement PORT
ENV PORT=7860

# Exposer le port utilisé par l'application Gradio
EXPOSE 7860

# Lancer l’application Gradio avec uv
CMD ["uv", "run", "python", "main.py"]