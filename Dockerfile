# ============================================
# Digital Human Clone Dashboard - Dockerfile
# ============================================
# Base: NVIDIA CUDA 12.4.1 with cuDNN (matches host CUDA 12.6 driver)
# 
# Build: docker-compose build
# Run:   docker-compose up
# ============================================

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ============================================
# SYSTEM DEPENDENCIES
# ============================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python and build tools
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    # Build essentials for llama-cpp-python
    cmake \
    build-essential \
    gcc \
    g++ \
    ninja-build \
    # Audio processing
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    # Git for installing packages from repos
    git \
    # Misc utilities
    curl \
    wget \
    ca-certificates \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && python -m pip install --upgrade pip setuptools wheel

# ============================================
# CUDA ENVIRONMENT VARIABLES
# ============================================

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Verify CUDA is available
RUN nvcc --version

# ============================================
# WORKING DIRECTORY
# ============================================

WORKDIR /app

# ============================================
# PYTHON DEPENDENCIES - Stage 1: PyTorch
# ============================================

# Install PyTorch with CUDA 12.4 support FIRST
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    torchaudio==2.4.1 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# ============================================
# PYTHON DEPENDENCIES - Stage 2: Coqui TTS
# ============================================

# Install Coqui TTS (has specific numpy requirements)
RUN pip install --no-cache-dir TTS==0.22.0

# ============================================
# PYTHON DEPENDENCIES - Stage 3: llama-cpp-python
# ============================================

# Install llama-cpp-python WITH CUDA support
# Using cuBLAS backend for NVIDIA GPUs
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=all-major"
ENV FORCE_CMAKE=1
ENV LLAMA_CUDA=1

# First install cmake and scikit-build-core requirements
RUN pip install --no-cache-dir \
    scikit-build-core \
    cmake \
    ninja

# Build llama-cpp-python from source with CUDA
# Using a known working version
RUN pip install --no-cache-dir \
    llama-cpp-python==0.2.90 \
    --force-reinstall \
    --no-binary llama-cpp-python \
    --verbose

# ============================================
# PYTHON DEPENDENCIES - Stage 4: Other packages
# ============================================

RUN pip install --no-cache-dir \
    # Whisper for transcription
    faster-whisper==1.0.3 \
    # ChromaDB for vector storage
    chromadb>=0.5.0 \
    # Audio processing
    pydub \
    ffmpeg-python \
    soundfile \
    scipy \
    # UI
    gradio>=4.40.0 \
    # Data validation
    pydantic>=2.0.0 \
    # NLP
    transformers>=4.40.0 \
    sentence-transformers>=3.0.0

# ============================================
# APPLICATION CODE
# ============================================

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/profiles /app/models /app/output /app/temp_audio

# ============================================
# RUNTIME CONFIGURATION
# ============================================

# Gradio server settings
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Coqui TOS agreement
ENV COQUI_TOS_AGREED=1

# Expose port
EXPOSE 7860

# ============================================
# ENTRYPOINT
# ============================================

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the application
CMD ["python", "app.py"]
