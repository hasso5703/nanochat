#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Configuration "Gaming PC" par Hasan Basbunar
# Adapté pour 1 seul GPU (RTX 3060/4070/4090)
# Modèle réduit à d12 (approx 125M paramètres) pour des temps d'entraînement réalistes.

# -----------------------------------------------------------------------------
# 1. Configuration Matérielle
# -----------------------------------------------------------------------------
# Réduisez ceci si vous avez une erreur "CUDA out of memory".
# 8GB VRAM -> 4
# 12GB VRAM -> 8
# 24GB VRAM -> 16 ou 32
DEVICE_BATCH_SIZE=8

# Profondeur du modèle (d12 = style GPT-2 Small).
# Le speedrun original est d20 (plus gros), mais prendrait une semaine sur un seul GPU.
MODEL_DEPTH=12 

# Setup des dossiers
export OMP_NUM_THREADS=8
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# 2. Installation (uv & venv)
# -----------------------------------------------------------------------------
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
# Installation des dépendances GPU
uv sync --extra gpu --prerelease=allow
source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# Reset du rapport
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# 3. Tokenizer & Data
# -----------------------------------------------------------------------------
# Téléchargement initial (8 shards = ~800MB)
python -m nanochat.dataset -n 8

# Téléchargement en arrière-plan.
# Pour d12 (125M params), on a besoin de ~2.5B tokens selon Chinchilla.
# Cela correspond à environ 50 shards de données.
python -m nanochat.dataset -n 50 &
DATASET_DOWNLOAD_PID=$!

# Entraînement du tokenizer (rapide, sur CPU)
python -m scripts.tok_train --max-chars=2000000000 --vocab-size=65536
python -m scripts.tok_eval

echo "Attente de la fin du téléchargement des données..."
wait $DATASET_DOWNLOAD_PID

# -----------------------------------------------------------------------------
# 4. Base Model (Pretraining) - L'étape la plus longue
# -----------------------------------------------------------------------------
# NOTE: Remplacement de 'torchrun' par 'python' pour le monocarte.
# Le code gère automatiquement l'accumulation de gradient.

echo "Démarrage du Pretraining (Ceci va prendre du temps)..."
python -m scripts.base_train \
    --depth=$MODEL_DEPTH \
    --target-param-data-ratio=20 \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --run=$WANDB_RUN

# Évaluation post-training
python -m scripts.base_loss --device-batch-size=$DEVICE_BATCH_SIZE
python -m scripts.base_eval

# -----------------------------------------------------------------------------
# 5. Midtraining (Apprendre à discuter)
# -----------------------------------------------------------------------------
# Téléchargement des données de personnalité
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

python -m scripts.mid_train \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --run=$WANDB_RUN

python -m scripts.chat_eval -i mid

# -----------------------------------------------------------------------------
# 6. Supervised Finetuning (Affinement)
# -----------------------------------------------------------------------------
python -m scripts.chat_sft \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --run=$WANDB_RUN

python -m scripts.chat_eval -i sft

# -----------------------------------------------------------------------------
# 7. Rapport & Lancement
# -----------------------------------------------------------------------------
python -m nanochat.report generate

echo "Terminé ! Pour discuter avec ton modèle :"
echo "python -m scripts.chat_web"