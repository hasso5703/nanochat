# Construction de l'image (environ 2-3 minutes car l'image de base est en cache sur le DGX)
docker build -f Dockerfile.dgx-spark -t nanochat:gb10 .

# Lancement interactif (avec accès total aux bus NVLink de la GB10)
docker run --gpus all -it --rm \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd)/data:/workspace/data \
    nanochat:gb10