#!/usr/bin/env bash
set -e

# install uv if not found
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# virtual environment
echo "Creating virtual environment (.venv)..."
uv venv
source .venv/bin/activate

# install torch
echo "Installing PyTorch with CUDA 13.2 support..."
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu132

# install project
echo "Installing eic-speech-nemo..."
uv pip install -e .

echo ""
echo "Setup complete!"
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
