"""
Digital Human Clone Dashboard
=============================
Main Gradio Application with Gatekeeper UI Flow

This is the ONLY file that imports Gradio.
All core logic lives in the modules/ directory.

Features:
- Gatekeeper Flow: Train/Chat tabs hidden until profile is active
- Status Banner: Shows current active profile
- One-Click Training: Single button to run entire pipeline
- Progress Feedback: Real-time updates during training

Author: AI Solutions Architect
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr

# Import core modules (no Gradio dependencies)
from modules.profile_manager import ProfileManager
from modules.ear import Ear, get_ear
from modules.brain import Brain, get_brain
from modules.mouth import Mouth, get_mouth
from modules.orchestrator import Orchestrator, get_orchestrator, PipelineProgress

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Path to your Llama 3 GGUF model
# Download from: https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF
LLAMA_MODEL_PATH = os.getenv(
    "LLAMA_MODEL_PATH",
    "models/Llama-3-8B-Instruct-v0.1.Q5_K_M.gguf"
)

# Whisper model size (tiny, base, small, medium, large-v2, large-v3)
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")

# Device settings
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "float32"

print(f"🚀 Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
print(f"📁 Llama Model: {LLAMA_MODEL_PATH}")
print(f"🎤 Whisper Model: {WHISPER_MODEL_SIZE}")

# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCES
# ═══════════════════════════════════════════════════════════════════════════

profile_manager = ProfileManager()

# Lazy-loaded module instances
_ear: Optional[Ear] = None
_mouth: Optional[Mouth] = None
_brain_instances: dict = {}


def get_ear_instance() -> Ear:
    """Get the Ear (transcription) instance."""
    global _ear
    if _ear is None:
        _ear = Ear(
            model_size=WHISPER_MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )
    return _ear


def get_mouth_instance() -> Mouth:
    """Get the Mouth (TTS) instance."""
    global _mouth
    if _mouth is None:
        _mouth = Mouth(device=DEVICE, output_dir="output")
    return _mouth


def get_brain_instance(profile_name: str) -> Brain:
    """Get a Brain instance for a specific profile."""
    global _brain_instances
    
    if profile_name not in _brain_instances:
        vector_db_dir = str(profile_manager.get_vector_db_dir(profile_name))
        _brain_instances[profile_name] = Brain(
            collection_name=f"persona_{profile_name}",
            persist_directory=vector_db_dir,
            model_path=LLAMA_MODEL_PATH if os.path.exists(LLAMA_MODEL_PATH) else None,
            n_ctx=4096,
            n_gpu_layers=-1
        )
    
    return _brain_instances[profile_name]


def get_orchestrator_instance() -> Orchestrator:
    """Get the Orchestrator instance."""
    return get_orchestrator(
        profile_manager,
        ear_instance=None,  # Lazy load
        brain_factory=get_brain_instance,
        mouth_instance=None  # Lazy load
    )


# ═══════════════════════════════════════════════════════════════════════════
# UI CALLBACK FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_status_banner(profile_name: Optional[str]) -> str:
    """Generate the status banner text."""
    if not profile_name:
        return "## 🔒 Current Profile: **[None]** - Select or create a profile to begin"
    return f"## ✅ Current Profile: **{profile_name}**"


def refresh_profiles() -> gr.Dropdown:
    """Refresh the profile dropdown choices."""
    profiles = profile_manager.list_profiles()
    return gr.Dropdown(choices=profiles, value=profiles[0] if profiles else None)


def create_new_profile(
    name: str, 
    description: str
) -> Tuple[str, gr.Dropdown, str, gr.Tab, gr.Tab]:
    """
    Create a new profile and activate Gatekeeper.
    
    Returns:
        Tuple of (status, dropdown, banner, train_tab_update, chat_tab_update)
    """
    if not name or not name.strip():
        return (
            "⚠️ Please enter a profile name.",
            gr.Dropdown(),
            get_status_banner(None),
            gr.Tab(visible=False),
            gr.Tab(visible=False)
        )
    
    success, message = profile_manager.create_profile(name.strip(), description)
    profiles = profile_manager.list_profiles()
    
    if success:
        # Profile created - unlock tabs
        selected_profile = name.strip()
        return (
            message,
            gr.Dropdown(choices=profiles, value=selected_profile),
            get_status_banner(selected_profile),
            gr.Tab(visible=True),
            gr.Tab(visible=True)
        )
    else:
        # Failed - keep tabs locked
        return (
            message,
            gr.Dropdown(choices=profiles, value=profiles[0] if profiles else None),
            get_status_banner(None),
            gr.Tab(visible=False),
            gr.Tab(visible=False)
        )


def select_profile(
    profile_name: Optional[str]
) -> Tuple[str, str, gr.Tab, gr.Tab, List[Tuple[str, str]]]:
    """
    Handle profile selection - activate Gatekeeper.
    
    Returns:
        Tuple of (profile_info, banner, train_tab_update, chat_tab_update, chat_history)
    """
    if not profile_name:
        return (
            "No profile selected.",
            get_status_banner(None),
            gr.Tab(visible=False),
            gr.Tab(visible=False),
            []
        )
    
    # Profile selected - unlock tabs
    profile_info = get_profile_info(profile_name)
    chat_history = load_chat_history(profile_name)
    
    return (
        profile_info,
        get_status_banner(profile_name),
        gr.Tab(visible=True),
        gr.Tab(visible=True),
        chat_history
    )


def get_profile_info(profile_name: str) -> str:
    """Get profile metadata and status."""
    if not profile_name:
        return "No profile selected."
    
    metadata = profile_manager.get_profile_metadata(profile_name)
    if not metadata:
        return "Profile not found."
    
    audio_files = profile_manager.get_audio_files(profile_name)
    transcripts_path = profile_manager.get_transcripts_path(profile_name)
    has_transcripts = transcripts_path.exists() and transcripts_path.stat().st_size > 0
    
    # Get vector db stats
    try:
        brain = get_brain_instance(profile_name)
        stats = brain.get_collection_stats()
        vector_count = stats.get("document_count", 0)
    except Exception:
        vector_count = 0
    
    info = f"""
## 📁 Profile: {metadata.name}

**Description:** {metadata.description or 'No description'}

**Created:** {metadata.created_at[:10]}

---

### Status:
- 🎵 **Audio Files:** {len(audio_files)}
- 📝 **Transcripts:** {'✅ Available' if has_transcripts else '❌ Not yet'}
- 🧠 **Vector Chunks:** {vector_count}

---

### Training Status:
{'✅ Ready for chat!' if vector_count > 0 else '⏳ Upload audio and click Start Training'}
"""
    return info


def upload_audio_files(profile_name: str, files: List[str]) -> str:
    """Save uploaded audio files to profile (without processing)."""
    if not profile_name:
        return "⚠️ Please select a profile first."
    
    if not files:
        return "⚠️ No files uploaded."
    
    saved = []
    errors = []
    
    for file_path in files:
        success, result = profile_manager.save_uploaded_audio(profile_name, file_path)
        if success:
            saved.append(Path(result).name)
        else:
            errors.append(result)
    
    message = f"✅ Saved {len(saved)} files to profile '{profile_name}'."
    if errors:
        message += f"\n⚠️ Errors: {'; '.join(errors)}"
    
    message += "\n\n📌 Click **Start Training** to process these files."
    
    return message


def run_one_click_training(
    profile_name: str, 
    files: Optional[List[str]], 
    progress=gr.Progress()
) -> Tuple[str, str]:
    """
    Run the complete one-click training pipeline.
    
    Args:
        profile_name: Active profile name
        files: Optional new files to add
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (training_status, profile_info)
    """
    if not profile_name:
        return "⚠️ Please select a profile first.", get_profile_info(None)
    
    # Check if we have audio files
    existing_files = profile_manager.get_audio_files(profile_name)
    new_files = files if files else []
    
    if not existing_files and not new_files:
        return (
            "⚠️ No audio files found. Please upload audio files first.",
            get_profile_info(profile_name)
        )
    
    # Create orchestrator
    orchestrator = get_orchestrator_instance()
    
    # Determine if we need to run ingest
    skip_ingest = len(new_files) == 0
    
    # Run pipeline with progress updates
    try:
        pipeline = orchestrator.run_training_pipeline(
            profile_name=profile_name,
            audio_files=new_files if new_files else None,
            skip_ingest=skip_ingest
        )
        
        final_result = None
        
        for step_progress in pipeline:
            # Update Gradio progress bar
            progress_value = step_progress.current_step / step_progress.total_steps
            progress(
                progress_value, 
                desc=f"Step {step_progress.current_step}/{step_progress.total_steps}: {step_progress.step_description}"
            )
            
            if step_progress.is_error:
                # Pipeline failed - return error
                return (
                    f"❌ Pipeline failed at step {step_progress.current_step}: {step_progress.error_message}",
                    get_profile_info(profile_name)
                )
            
            if step_progress.is_complete:
                break
        
        # Try to get the return value
        try:
            # The generator has completed, get the result
            pass
        except StopIteration as e:
            final_result = e.value
        
        # Build success message
        status_msg = f"""
## ✅ Training Complete!

Your digital human **{profile_name}** is ready.

### Pipeline Summary:
- 🎵 Audio files processed
- 📝 Speech transcribed with Whisper AI
- 🧠 Transcripts indexed for semantic search
- 🗣️ Voice samples prepared for cloning

---

**Next Step:** Go to the **Chat** tab to interact with your digital human!
"""
        return status_msg, get_profile_info(profile_name)
        
    except Exception as e:
        return (
            f"❌ Training failed: {str(e)}",
            get_profile_info(profile_name)
        )


def chat_respond(
    profile_name: str,
    user_message: str,
    chat_history: List[Tuple[str, str]],
    language: str
) -> Tuple[List[Tuple[str, str]], str, Optional[str]]:
    """
    Process a chat message: Brain thinks, Mouth speaks.
    
    Returns:
        Updated chat history, status message, audio path
    """
    if not profile_name:
        return chat_history, "⚠️ Please select a profile first.", None
    
    if not user_message.strip():
        return chat_history, "⚠️ Please enter a message.", None
    
    # Check if Llama model exists
    if not os.path.exists(LLAMA_MODEL_PATH):
        # Fallback: Just generate audio without Brain
        mouth = get_mouth_instance()
        audio_result = mouth.speak(
            text=user_message,
            profile_audio_dir=str(profile_manager.get_audio_dir(profile_name)),
            language=language
        )
        
        if audio_result.success:
            # Add user message and echo it back
            chat_history.append((user_message, f"[Audio generated] {user_message}"))
            profile_manager.save_chat_message(profile_name, "user", user_message)
            profile_manager.save_chat_message(
                profile_name, "assistant", user_message, 
                audio_result.audio_path
            )
            return chat_history, "✅ Audio generated (no LLM available - echoed input)", audio_result.audio_path
        else:
            return chat_history, audio_result.message, None
    
    try:
        # Get Brain response
        brain = get_brain_instance(profile_name)
        gen_result = brain.generate(
            user_input=user_message,
            n_context_chunks=5,
            max_tokens=256,
            temperature=0.7
        )
        
        if not gen_result.success:
            return chat_history, gen_result.message, None
        
        response_text = gen_result.response
        
        # Generate audio with Mouth
        mouth = get_mouth_instance()
        audio_result = mouth.speak(
            text=response_text,
            profile_audio_dir=str(profile_manager.get_audio_dir(profile_name)),
            language=language
        )
        
        # Update chat history
        chat_history.append((user_message, response_text))
        
        # Save to profile history
        profile_manager.save_chat_message(profile_name, "user", user_message)
        profile_manager.save_chat_message(
            profile_name, "assistant", response_text,
            audio_result.audio_path if audio_result.success else None
        )
        
        status = gen_result.message
        if audio_result.success:
            status += " | " + audio_result.message
        else:
            status += " | ⚠️ Audio generation failed."
        
        return chat_history, status, audio_result.audio_path if audio_result.success else None
        
    except Exception as e:
        return chat_history, f"❌ Error: {str(e)}", None


def clear_chat(profile_name: str) -> Tuple[List, str]:
    """Clear chat history."""
    if profile_name:
        profile_manager.clear_chat_history(profile_name)
    return [], "Chat cleared."


def load_chat_history(profile_name: str) -> List[Tuple[str, str]]:
    """Load existing chat history for a profile."""
    if not profile_name:
        return []
    
    history = profile_manager.load_chat_history(profile_name)
    
    # Convert to Gradio format
    pairs = []
    user_msg = None
    for msg in history.messages:
        if msg.role == "user":
            user_msg = msg.content
        elif msg.role == "assistant" and user_msg:
            pairs.append((user_msg, msg.content))
            user_msg = None
    
    return pairs


# ═══════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════

# Custom CSS for premium aesthetics
custom_css = """
/* Status Banner */
.status-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 12px;
    padding: 16px 24px;
    border: 1px solid #0f3460;
    margin-bottom: 20px;
}

.status-banner.locked {
    background: linear-gradient(135deg, #2d1b1b 0%, #3d1f1f 100%);
    border-color: #5c2e2e;
}

.status-banner.unlocked {
    background: linear-gradient(135deg, #1b2d1b 0%, #1f3d2a 100%);
    border-color: #2e5c3a;
}

/* Profile Info Card */
.profile-info {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #0f3460;
}

/* Status Box */
.status-box {
    font-family: 'Consolas', 'Monaco', monospace;
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px;
}

/* Training Progress */
.training-status {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #30363d;
}

/* Container */
.gradio-container {
    max-width: 1400px !important;
}

/* Tab Navigation */
.tab-nav button {
    font-size: 16px !important;
    padding: 12px 24px !important;
}

/* Disabled tabs styling */
.tab-nav button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

/* Primary Button Enhancement */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    transition: transform 0.2s ease !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
}

/* Start Training Button */
.start-training-btn {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    font-size: 18px !important;
    padding: 16px 32px !important;
}
"""

# Build the Gradio interface
# Note: In Gradio 6.x, theme is applied via constructor, css via head param
with gr.Blocks(
    title="🎭 Digital Human Clone Dashboard"
) as app:
    # Apply custom CSS via head
    gr.HTML(f"<style>{custom_css}</style>")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STATUS BANNER (Gatekeeper Indicator)
    # ─────────────────────────────────────────────────────────────────────────
    status_banner = gr.Markdown(
        value=get_status_banner(None),
        elem_classes=["status-banner"]
    )
    
    # Header
    gr.Markdown("""
    # 🎭 Digital Human Clone Dashboard
    
    **Create AI-powered personas that speak, think, and respond in any voice.**
    
    ---
    """)
    
    with gr.Row():
        # ─────────────────────────────────────────────────────────────────────
        # SIDEBAR: Profile Management (Always Visible)
        # ─────────────────────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("## 📂 Profile Manager")
            
            with gr.Group():
                profile_dropdown = gr.Dropdown(
                    label="Select Profile",
                    choices=profile_manager.list_profiles(),
                    value=None,  # Start with no profile
                    interactive=True
                )
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
            
            gr.Markdown("### Create New Profile")
            with gr.Group():
                new_profile_name = gr.Textbox(
                    label="Profile Name",
                    placeholder="e.g., Comedian_Joe"
                )
                new_profile_desc = gr.Textbox(
                    label="Description (optional)",
                    placeholder="Famous standup comedian..."
                )
                create_btn = gr.Button("➕ Create Profile", variant="primary")
                create_status = gr.Textbox(label="Status", interactive=False, lines=1)
            
            # Profile info display
            profile_info_display = gr.Markdown(
                value="Select or create a profile to begin.",
                elem_classes=["profile-info"]
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # MAIN CONTENT: Tabs (Gatekeeper Controlled)
        # ─────────────────────────────────────────────────────────────────────
        with gr.Column(scale=3):
            with gr.Tabs() as tabs:
                
                # ═══════════════════════════════════════════════════════════════
                # TAB 1: TRAIN (Hidden by default - Gatekeeper)
                # ═══════════════════════════════════════════════════════════════
                with gr.Tab("🎓 Train", id="train", visible=False) as train_tab:
                    gr.Markdown("""
                    ### 📤 Upload Voice Samples
                    Upload clean audio files of the person you want to clone.
                    **Tips:** Use files with clear speech, minimal background noise, 5-30 seconds each.
                    """)
                    
                    with gr.Row():
                        audio_upload = gr.File(
                            label="Upload Audio Files (MP3/WAV/OGG)",
                            file_count="multiple",
                            type="filepath",
                            file_types=["audio"]
                        )
                    
                    upload_status = gr.Textbox(
                        label="Upload Status",
                        interactive=False,
                        elem_classes=["status-box"]
                    )
                    
                    gr.Markdown("---")
                    
                    # ONE-CLICK TRAINING
                    gr.Markdown("""
                    ### 🚀 One-Click Training
                    Click the button below to run the complete training pipeline:
                    
                    1. **Ingest** → Save audio files
                    2. **Ear** → Transcribe with Whisper AI  
                    3. **Brain** → Index for semantic search
                    4. **Mouth** → Prepare for voice cloning
                    """)
                    
                    start_training_btn = gr.Button(
                        "🎯 Start Training",
                        variant="primary",
                        size="lg",
                        elem_classes=["start-training-btn"]
                    )
                    
                    training_status = gr.Markdown(
                        value="*Ready to train. Upload audio files and click Start Training.*",
                        elem_classes=["training-status"]
                    )
                    
                    # Progress indicator (built into Gradio)
                    gr.Markdown("""
                    ---
                    **What happens:**
                    - **Transcribe (Ear):** Uses Whisper AI to convert speech to text
                    - **Vectorize (Brain):** Indexes the text for semantic search (RAG)
                    - **Preprocess (Mouth):** Prepares audio for voice cloning
                    
                    After training, go to the **Chat** tab to interact!
                    """)
                
                # ═══════════════════════════════════════════════════════════════
                # TAB 2: CHAT (Hidden by default - Gatekeeper)
                # ═══════════════════════════════════════════════════════════════
                with gr.Tab("💬 Chat", id="chat", visible=False) as chat_tab:
                    gr.Markdown("""
                    ### Chat with Your Digital Human
                    The AI will respond in the persona's style and speak with their voice.
                    """)
                    
                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=400
                    )
                    
                    with gr.Row():
                        chat_input = gr.Textbox(
                            label="Your Message",
                            placeholder="Type something to say...",
                            scale=4,
                            lines=2
                        )
                        language_dropdown = gr.Dropdown(
                            label="Language",
                            choices=Mouth.SUPPORTED_LANGUAGES,
                            value="en",
                            scale=1
                        )
                    
                    with gr.Row():
                        send_btn = gr.Button("🎤 Send & Speak", variant="primary", scale=2)
                        clear_btn = gr.Button("🗑️ Clear Chat", scale=1)
                    
                    # Audio output with autoplay
                    audio_output = gr.Audio(
                        label="🔊 Voice Response",
                        autoplay=True,
                        type="filepath"
                    )
                    
                    chat_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        elem_classes=["status-box"]
                    )
                    
                    # LLM Warning
                    if not os.path.exists(LLAMA_MODEL_PATH):
                        gr.Markdown(f"""
                        ⚠️ **Llama model not found at:** `{LLAMA_MODEL_PATH}`
                        
                        Without the LLM, chat will only generate voice audio of your input.
                        
                        To enable AI responses:
                        1. Download a Llama 3 GGUF model from HuggingFace
                        2. Place it in the `models/` directory
                        3. Set `LLAMA_MODEL_PATH` environment variable
                        """)
    
    # ═════════════════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ═════════════════════════════════════════════════════════════════════════
    
    # Profile management - Refresh
    refresh_btn.click(
        fn=refresh_profiles,
        outputs=[profile_dropdown]
    )
    
    # Profile creation - Gatekeeper activation
    create_btn.click(
        fn=create_new_profile,
        inputs=[new_profile_name, new_profile_desc],
        outputs=[create_status, profile_dropdown, status_banner, train_tab, chat_tab]
    ).then(
        fn=get_profile_info,
        inputs=[profile_dropdown],
        outputs=[profile_info_display]
    )
    
    # Profile selection - Gatekeeper activation
    profile_dropdown.change(
        fn=select_profile,
        inputs=[profile_dropdown],
        outputs=[profile_info_display, status_banner, train_tab, chat_tab, chatbot]
    )
    
    # Audio upload (just saves, doesn't process)
    audio_upload.change(
        fn=upload_audio_files,
        inputs=[profile_dropdown, audio_upload],
        outputs=[upload_status]
    ).then(
        fn=get_profile_info,
        inputs=[profile_dropdown],
        outputs=[profile_info_display]
    )
    
    # ONE-CLICK TRAINING - Main action
    start_training_btn.click(
        fn=run_one_click_training,
        inputs=[profile_dropdown, audio_upload],
        outputs=[training_status, profile_info_display]
    )
    
    # Chat actions
    send_btn.click(
        fn=chat_respond,
        inputs=[profile_dropdown, chat_input, chatbot, language_dropdown],
        outputs=[chatbot, chat_status, audio_output]
    ).then(
        fn=lambda: "",
        outputs=[chat_input]
    )
    
    # Also send on Enter key
    chat_input.submit(
        fn=chat_respond,
        inputs=[profile_dropdown, chat_input, chatbot, language_dropdown],
        outputs=[chatbot, chat_status, audio_output]
    ).then(
        fn=lambda: "",
        outputs=[chat_input]
    )
    
    clear_btn.click(
        fn=clear_chat,
        inputs=[profile_dropdown],
        outputs=[chatbot, chat_status]
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎭 Digital Human Clone Dashboard")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("data/profiles", exist_ok=True)
    
    print(f"\n📁 Data directory: data/profiles/")
    print(f"📁 Models directory: models/")
    print(f"📁 Output directory: output/")
    
    if not os.path.exists(LLAMA_MODEL_PATH):
        print(f"\n⚠️  Llama model not found at: {LLAMA_MODEL_PATH}")
        print("   Chat will work in voice-only mode (no AI responses)")
        print("   Download Llama 3 GGUF from HuggingFace to enable AI chat")
    
    print("\n🚀 Starting server...")
    print("=" * 60 + "\n")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
