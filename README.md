# 🎙️ AI Voice Cloning Dashboard

A powerful AI-powered voice cloning application built with **Coqui TTS (XTTS v2)** and **Gradio**. Upload voice samples to clone any voice and generate new speech in 16+ languages.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-Supported-brightgreen.svg)

---

## ✨ Features

- **Zero-Shot Voice Cloning** – Clone voices from just a few audio samples
- **Multi-Language Support** – Generate speech in 16 languages (English, Spanish, French, German, Arabic, Chinese, Japanese, and more)
- **GPU Acceleration** – Automatic CUDA detection for fast inference
- **Simple Web UI** – Easy-to-use Gradio interface with two-tab workflow
- **Multiple Audio Formats** – Supports MP3, WAV, and other common formats
- **Shareable Links** – Built-in support for public sharing via Gradio

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- FFmpeg installed on your system
- NVIDIA GPU with CUDA (optional, but recommended for faster processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/alsaeed3/voice_model_training.git
   cd voice_model_training
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install FFmpeg** (if not already installed)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg

   # macOS
   brew install ffmpeg

   # Windows (using Chocolatey)
   choco install ffmpeg
   ```

### Running the Application

```bash
python app.py
```

The application will start and provide:
- A local URL (e.g., `http://127.0.0.1:7860`)
- A public shareable URL (via Gradio's share feature)

---

## 📖 How to Use

### Step 1: Upload Voice Samples
1. Navigate to the **"Upload & Train"** tab
2. Upload clean MP3 or WAV files of the target voice
   - Use clips with clear speech and minimal background noise
   - Multiple samples improve voice quality
3. Click **"Process Voice Files"** to prepare the audio

### Step 2: Generate Speech
1. Switch to the **"Generate Audio"** tab
2. Enter the text you want spoken
3. Select the target language
4. Click **"Generate Audio"** to create the cloned voice output

---

## 🌍 Supported Languages

| Language | Code | Language | Code |
|----------|------|----------|------|
| English | `en` | Polish | `pl` |
| Spanish | `es` | Turkish | `tr` |
| French | `fr` | Russian | `ru` |
| German | `de` | Dutch | `nl` |
| Italian | `it` | Czech | `cs` |
| Portuguese | `pt` | Hungarian | `hu` |
| Arabic | `ar` | Chinese (Simplified) | `zh-cn` |
| Japanese | `ja` | Korean | `ko` |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Voice Cloning Model | [Coqui TTS XTTS v2](https://github.com/coqui-ai/TTS) |
| Web Interface | [Gradio](https://gradio.app/) |
| Deep Learning | [PyTorch](https://pytorch.org/) |
| Audio Processing | [PyDub](https://github.com/jiaaro/pydub) + FFmpeg |

---

## ⚙️ Configuration

### GPU vs CPU

The application automatically detects CUDA availability:
- **With GPU**: Faster inference, recommended for production use
- **Without GPU**: Runs on CPU (slower but fully functional)

### Environment Variables

| Variable | Description |
|----------|-------------|
| `COQUI_TOS_AGREED` | Set to `1` automatically to accept Coqui TOS |

---

## 📁 Project Structure

```
voice_model_training/
├── app.py              # Main application with Gradio UI
├── requirements.txt    # Python dependencies
├── LICENSE             # MIT License
├── README.md           # This file
├── temp_audio/         # Temporary processed audio files (auto-created)
└── output_cloned.wav   # Generated audio output (auto-created)
```

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Model download fails | Ensure stable internet connection; model is ~2GB |
| Audio processing error | Verify FFmpeg is installed and in PATH |
| CUDA out of memory | Reduce batch size or use shorter audio clips |
| Generation quality poor | Use cleaner audio samples with less noise |

### Tips for Best Results

1. **Clean Audio**: Use samples without background music or noise
2. **Multiple Samples**: 3-5 diverse samples improve voice quality
3. **Sample Length**: 10-30 seconds per sample works best
4. **Consistent Voice**: Use samples from the same speaker only

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Coqui AI](https://coqui.ai/) for the incredible XTTS v2 model
- [Gradio](https://gradio.app/) for the intuitive web UI framework
- [PyTorch](https://pytorch.org/) for the deep learning backbone

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

<p align="center">
  Made with ❤️ by Ali Saeed
</p>