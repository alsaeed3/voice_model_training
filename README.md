# 🎭 Digital Human Clone Dashboard

A modular, production-ready AI system for creating digital human clones with **Voice Cloning (Mouth)**, **Transcription (Ear)**, and **Style/Persona (Brain)** capabilities. Features a "Gatekeeper" UI flow and one-click training pipeline.

---

## 🏗️ Architecture

```
voice_model_training/
├── app.py                          # Main Gradio UI (ONLY file with Gradio imports)
├── requirements.txt                # Python dependencies (voice cloning)
├── requirements-brain.txt          # AI brain dependencies (LLM + RAG)
├── setup.sh                        # Installation script (handles CUDA builds)
├── Dockerfile                      # Container configuration
├── docker-compose.yml              # Orchestration with volume mounts
├── models/                         # Llama 3 GGUF models (user-provided)
│   └── *.gguf                      # Download your own model
├── output/                         # Generated audio files
├── temp_audio/                     # Temporary processed audio
├── data/
│   └── profiles/                   # Profile data (portable folders)
│       └── {profile_name}/
│           ├── audio/              # Uploaded MP3s/WAVs
│           ├── vector_db/          # ChromaDB persistence
│           ├── transcripts.txt     # Raw text from Whisper
│           ├── chat_history.json   # Conversation history
│           └── metadata.json       # Profile metadata
└── modules/                        # Core logic (NO Gradio dependencies)
    ├── __init__.py
    ├── ear.py                      # Transcription (faster-whisper)
    ├── brain.py                    # RAG (ChromaDB + Llama 3)
    ├── mouth.py                    # Voice synthesis (Coqui XTTS v2)
    ├── orchestrator.py             # Training pipeline orchestration
    └── profile_manager.py          # Profile folder management
```

---

## 🧩 Modules

### 👂 Ear (`modules/ear.py`)
- Uses **faster-whisper** for transcription
- Supports multiple model sizes: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`
- GPU-accelerated with CUDA
- Saves transcripts to profile folder

### 🧠 Brain (`modules/brain.py`)
- **ChromaDB** for vector storage and semantic search
- **llama-cpp-python** for Llama 3 inference
- RAG pipeline: Query → Search Context → Generate Response
- Persona-aware system prompts

### 👄 Mouth (`modules/mouth.py`)
- **Coqui TTS** with XTTS v2 for voice cloning
- Multi-language support (16+ languages)
- Automatic audio preprocessing

### 🎯 Orchestrator (`modules/orchestrator.py`)
- One-click training pipeline
- Coordinates Ear → Brain → Mouth workflow
- Handles profile initialization and state management

### 📂 Profile Manager (`modules/profile_manager.py`)
- File-system based storage (JSONs)
- Portable profile folders (copy/paste to share)
- Chat history management
- Pydantic validation

---

## ⚡ Quick Start

### Option 1: Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/voice_model_training.git
cd voice_model_training

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (recommended for CUDA users)
chmod +x setup.sh
./setup.sh

# Or manual install
pip install -r requirements.txt
pip install -r requirements-brain.txt
```

### Option 2: Docker (Recommended for Production)

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t digital-human .
docker run -p 7860:7860 --gpus all digital-human
```

### Download Llama 3 Model (Optional - for AI chat)

```bash
mkdir -p models
cd models

# Download from HuggingFace (example: Q5_K_M quantization)
wget https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf
```

### Run the Dashboard

```bash
# Local
python app.py

# With Docker
docker-compose up
```

Open `http://localhost:7860` in your browser.

---

## 🎯 Usage

### Creating a Profile

1. Enter a name in "Profile Name" (e.g., `Comedian_Joe`)
2. Click **Create Profile**
3. Select the profile from the dropdown

### Training a Voice (One-Click)

1. **Upload Audio Files**: Clean MP3/WAV files (5-30 seconds each)
2. Click **🚀 Train All**: Runs the complete pipeline:
   - Processes audio files
   - Transcribes with Whisper (Ear)
   - Vectorizes for RAG (Brain)
   - Prepares voice model (Mouth)

### Chatting

1. Go to the **Chat** tab
2. Type a message and click **Send & Speak**
3. The AI responds in the persona's style with their cloned voice
4. Audio auto-plays

---

## 🐳 Docker Configuration

### docker-compose.yml Features

- **GPU Support**: Automatic NVIDIA GPU passthrough
- **Volume Mounts**: Persistent data for models, profiles, and output
- **Health Checks**: Automatic container health monitoring
- **Resource Limits**: Configurable memory and CPU limits

### Environment Variables

```yaml
environment:
  - LLAMA_MODEL_PATH=/app/models/your-model.gguf
  - WHISPER_MODEL_SIZE=medium
  - CUDA_VISIBLE_DEVICES=0
```

---

## 🔧 Configuration

Set environment variables to customize:

```bash
# Llama model path
export LLAMA_MODEL_PATH="models/Llama-3-8B-Instruct-v0.1.Q5_K_M.gguf"

# Whisper model size (tiny, base, small, medium, large-v2, large-v3)
export WHISPER_MODEL_SIZE="medium"
```

---

## 📦 Dependencies

| Component | Library | Purpose |
|-----------|---------|---------|
| Voice Cloning | `TTS` (Coqui) | XTTS v2 voice synthesis |
| Transcription | `faster-whisper` | Speech-to-text |
| LLM | `llama-cpp-python` | Llama 3 inference |
| Vector Store | `chromadb` | Semantic search |
| Embeddings | `sentence-transformers` | Text embeddings |
| UI | `gradio` | Web interface |
| Validation | `pydantic` | Data models |

---

## 🚀 Deployment

The codebase is designed for easy migration:

### FastAPI Backend

The modules (`ear.py`, `brain.py`, `mouth.py`, `orchestrator.py`) have **no Gradio dependencies**. They take pure data as input and return pure data. Simply import them in your FastAPI routes.

```python
from fastapi import FastAPI
from modules.brain import get_brain
from modules.mouth import get_mouth

app = FastAPI()

@app.post("/chat")
async def chat(profile_name: str, message: str):
    brain = get_brain(profile_name, f"data/profiles/{profile_name}/vector_db")
    result = brain.generate(message)
    return {"response": result.response}
```

### Profile Portability

Profile folders are self-contained. Copy `data/profiles/Comedian_Joe/` to share it.

---

## ⚠️ Requirements

- **GPU**: NVIDIA GPU with CUDA support recommended
- **VRAM**: 8GB+ for medium Whisper + XTTS v2 + Llama 3 (8B Q4/Q5)
- **Python**: 3.10+
- **ffmpeg**: Required for audio processing

```bash
# Install ffmpeg (Ubuntu/Debian)
sudo apt install ffmpeg

# Install ffmpeg (macOS)
brew install ffmpeg
```

---

## 📁 Sharing Profiles

Profiles are fully portable:

```bash
# Export a profile
cp -r data/profiles/Comedian_Joe ~/Desktop/

# Import a profile
cp -r ~/Desktop/Comedian_Joe data/profiles/
```

---

## 🐛 Troubleshooting

### "Model failed to load"
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check VRAM usage with `nvidia-smi`

### "No Llama model found"
- Download a GGUF model and set `LLAMA_MODEL_PATH`
- The app works without it (voice-only mode)

### Transcription slow
- Use a smaller Whisper model: `export WHISPER_MODEL_SIZE="base"`

### Docker GPU Issues
- Ensure NVIDIA Container Toolkit is installed
- Verify with `docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

### Dependency Conflicts
- Use the provided `setup.sh` script for proper installation order
- Install numpy first, then other packages

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Coqui TTS](https://github.com/coqui-ai/TTS) - Voice cloning
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Transcription
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Llama inference
- [ChromaDB](https://github.com/chroma-core/chroma) - Vector store
- [Gradio](https://gradio.app/) - Web UI

---

## 👤 Author

**Ali Saeed** - [GitHub](https://github.com/alsaeed)