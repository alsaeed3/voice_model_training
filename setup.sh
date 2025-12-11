#!/bin/bash
# ============================================
# Digital Human Clone Dashboard - Setup Script
# ============================================
# Handles complex dependency installation

set -e

echo "🚀 Digital Human Clone Dashboard - Setup"
echo "=========================================="

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "📦 Python version: $PYTHON_VERSION"
echo ""

# Step 1: Install base requirements (TTS and core)
echo "📦 Step 1/4: Installing base requirements (TTS, PyTorch)..."
pip install --upgrade pip
pip install -r requirements.txt

# Step 2: Install faster-whisper and chromadb
echo ""
echo "📦 Step 2/4: Installing faster-whisper and ChromaDB..."
pip install faster-whisper
pip install chromadb
pip install sentence-transformers

# Step 3: Install llama-cpp-python with pre-built CUDA wheel
echo ""
echo "📦 Step 3/4: Installing llama-cpp-python (pre-built CUDA wheel)..."
# Use pre-built wheel from jllllll's repository (avoids CUDA toolkit requirement)
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Step 4: Fix numpy version to work with all packages
echo ""
echo "📦 Step 4/4: Fixing numpy compatibility..."
# Install a numpy version that works with most packages (ignore TTS's strict pin)
pip install "numpy>=1.24.0,<1.27.0" --force-reinstall

echo ""
echo "=========================================="
echo "✅ Installation complete!"
echo ""
echo "⚠️  Note: TTS will show a numpy warning but will work."
echo ""
echo "To run the dashboard:"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "Optional: Download Llama 3 model for AI chat:"
echo "  mkdir -p models"
echo "  # Recommended: 8B Q4_K_M (5.5GB)"
echo "  wget -P models/ 'https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf'"
