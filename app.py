import os
import sys
import torch
import torchaudio
import gradio as gr
from TTS.api import TTS
from pydub import AudioSegment

# --- Configuration & Setup ---
# Automatically agree to Coqui TOS so the model downloads without blocking
os.environ["COQUI_TOS_AGREED"] = "1"

# Check for CUDA (GPU) availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Device selected: {device}")
if device == "cuda":
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")

# --- Global Variables ---
# We load the model once to avoid reloading it on every click
tts_model = None

def load_model():
    """Loads the XTTS v2 model. Downloads it if not present."""
    global tts_model
    if tts_model is None:
        print("⏳ Loading XTTS v2 model... (This may take time on first run)")
        try:
            # Init TTS with the XTTS v2 model
            tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    return tts_model

# --- Processing Functions ---

def preprocess_audio(file_paths):
    """
    Takes a list of MP3/WAV files, converts them to WAV, 
    and verifies they are valid for training.
    """
    if not file_paths:
        return None, "No files uploaded."
    
    clean_files = []
    
    # Create a temp folder for converted wavs
    os.makedirs("temp_audio", exist_ok=True)
    
    for idx, file_path in enumerate(file_paths):
        try:
            # Convert to wav using pydub
            audio = AudioSegment.from_file(file_path)
            # Export as wav (16-bit PCM is standard for TTS)
            output_path = f"temp_audio/ref_{idx}.wav"
            audio.export(output_path, format="wav")
            clean_files.append(output_path)
        except Exception as e:
            print(f"⚠️ Could not process {file_path}: {e}")
    
    if not clean_files:
        return None, "Failed to process audio files."

    return clean_files, f"✅ Processed {len(clean_files)} audio files. Ready to clone."

def generate_voice(text, language, ref_audio_files):
    """
    Generates audio using the loaded model and reference audio files.
    """
    if not text.strip():
        return None, "⚠️ Please enter text to generate."
    if not ref_audio_files:
        return None, "⚠️ Please upload and process voice files in Tab 1 first."
    
    model = load_model()
    if not model:
        return None, "❌ Model failed to load."

    print(f"🗣️ Generating: '{text}' in {language}...")
    
    output_path = "output_cloned.wav"
    
    try:
        # XTTS generation
        # We pass the list of reference files directly. 
        # The model handles the embedding averaging internally or uses the first file 
        # depending on the specific API version, but passing the list is safer.
        model.tts_to_file(
            text=text,
            speaker_wav=ref_audio_files, # Pass the list of processed WAVs
            language=language,
            file_path=output_path
        )
        return output_path, "✅ Audio generated successfully!"
    except Exception as e:
        return None, f"❌ Error during generation: {str(e)}"

# --- Gradio UI Layout ---

with gr.Blocks(title="🎙️ Comedian Voice Cloner") as app:
    gr.Markdown("# 🎙️ AI Voice Cloning Dashboard")
    gr.Markdown("Upload MP3s of your comedian, process them, and then generate new audio.")

    # State variable to hold the processed audio paths across tabs
    ref_audio_state = gr.State(value=[])

    with gr.Tabs():
        # --- TAB 1: Training/Input ---
        with gr.Tab("1. Upload & Train"):
            gr.Markdown("### Step 1: Upload Voice Samples")
            gr.Markdown("Upload clean MP3/WAV files of the comedian (no background music).")
            
            file_input = gr.File(
                label="Upload Audio Files (MP3/WAV)", 
                file_count="multiple", 
                type="filepath"
            )
            process_btn = gr.Button("Process Voice Files", variant="primary")
            process_status = gr.Textbox(label="Status", interactive=False)
            
            # Button Action
            process_btn.click(
                fn=preprocess_audio,
                inputs=[file_input],
                outputs=[ref_audio_state, process_status]
            )

        # --- TAB 2: Inference ---
        with gr.Tab("2. Generate Audio"):
            gr.Markdown("### Step 2: Type & Speak")
            
            text_input = gr.Textbox(
                label="What should the comedian say?", 
                lines=3, 
                placeholder="I walked into a specific place today..."
            )
            
            # XTTS supports these languages
            lang_dropdown = gr.Dropdown(
                choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "ko", "hu"],
                value="en",
                label="Language"
            )
            
            generate_btn = gr.Button("Generate Audio", variant="primary")
            
            audio_output = gr.Audio(label="Generated Result")
            gen_status = gr.Textbox(label="Generation Status", interactive=False)
            
            # Button Action
            generate_btn.click(
                fn=generate_voice,
                inputs=[text_input, lang_dropdown, ref_audio_state],
                outputs=[audio_output, gen_status]
            )

# Load model on app launch (optional, makes startup slower but first generation faster)
# load_model()

if __name__ == "__main__":
    app.launch(share=True, server_name="0.0.0.0")
