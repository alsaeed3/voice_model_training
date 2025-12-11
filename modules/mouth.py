"""
Mouth Module - Voice Synthesis
==============================
Uses Coqui TTS (XTTS v2) for voice cloning and synthesis.
Generates audio from text using reference voice samples.

NO GRADIO DEPENDENCIES - Pure data in/out.
"""

import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ProcessingResult:
    """Result of audio preprocessing."""
    success: bool
    processed_files: List[str]
    message: str


@dataclass
class SynthesisResult:
    """Result of voice synthesis."""
    success: bool
    audio_path: str
    message: str


class Mouth:
    """
    Voice synthesis module using Coqui XTTS v2.
    
    Capabilities:
    - Load and preprocess reference audio
    - Generate speech in cloned voice
    - Multi-language support
    """
    
    SUPPORTED_LANGUAGES = [
        "en", "es", "fr", "de", "it", "pt", "pl", 
        "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "ko", "hu"
    ]
    
    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: Optional[str] = None,
        output_dir: str = "output"
    ):
        """
        Initialize the Mouth (TTS) module.
        
        Args:
            model_name: Coqui TTS model name
            device: "cuda" or "cpu" (auto-detect if None)
            output_dir: Directory for generated audio files
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect device if not specified
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self._model = None
        self._reference_files: List[str] = []
    
    def _load_model(self):
        """Lazy load the TTS model."""
        if self._model is None:
            try:
                # Set Coqui TOS agreement
                os.environ["COQUI_TOS_AGREED"] = "1"
                
                from TTS.api import TTS
                
                print(f"⏳ Loading TTS model on {self.device}...")
                self._model = TTS(self.model_name).to(self.device)
                print("✅ TTS model loaded successfully!")
                
            except Exception as e:
                print(f"❌ Failed to load TTS model: {e}")
                raise
        
        return self._model
    
    def preprocess_audio(
        self,
        file_paths: List[str],
        output_dir: Optional[str] = None
    ) -> ProcessingResult:
        """
        Convert audio files to WAV format suitable for voice cloning.
        
        Args:
            file_paths: List of input audio file paths
            output_dir: Directory to save processed files (uses temp if None)
            
        Returns:
            ProcessingResult with processed file paths
        """
        if not file_paths:
            return ProcessingResult(
                success=False,
                processed_files=[],
                message="No files provided."
            )
        
        from pydub import AudioSegment
        
        if output_dir is None:
            output_dir = self.output_dir / "temp_audio"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed = []
        errors = []
        
        for idx, file_path in enumerate(file_paths):
            try:
                # Load audio with pydub
                audio = AudioSegment.from_file(file_path)
                
                # Export as WAV (16-bit PCM)
                output_path = output_dir / f"ref_{idx}.wav"
                audio.export(
                    str(output_path),
                    format="wav",
                    parameters=["-ar", "22050", "-ac", "1"]  # 22.05kHz mono
                )
                processed.append(str(output_path))
                
            except Exception as e:
                errors.append(f"{Path(file_path).name}: {str(e)}")
        
        if not processed:
            return ProcessingResult(
                success=False,
                processed_files=[],
                message=f"Failed to process any files. Errors: {'; '.join(errors)}"
            )
        
        message = f"✅ Processed {len(processed)}/{len(file_paths)} audio files."
        if errors:
            message += f"\n⚠️ Errors: {'; '.join(errors)}"
        
        return ProcessingResult(
            success=True,
            processed_files=processed,
            message=message
        )
    
    def set_reference_audio(self, file_paths: List[str]) -> tuple[bool, str]:
        """
        Set the reference audio files for voice cloning.
        
        Args:
            file_paths: List of WAV file paths to use as reference
            
        Returns:
            Tuple of (success, message)
        """
        valid_files = []
        
        for path in file_paths:
            if os.path.exists(path):
                valid_files.append(path)
            else:
                print(f"⚠️ Reference file not found: {path}")
        
        if not valid_files:
            return False, "No valid reference audio files found."
        
        self._reference_files = valid_files
        return True, f"✅ Set {len(valid_files)} reference audio files."
    
    def synthesize(
        self,
        text: str,
        language: str = "en",
        reference_files: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> SynthesisResult:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            language: Language code (e.g., "en", "es")
            reference_files: Reference audio for voice cloning (uses stored if None)
            output_path: Path for output audio (auto-generated if None)
            
        Returns:
            SynthesisResult with audio path
        """
        if not text.strip():
            return SynthesisResult(
                success=False,
                audio_path="",
                message="⚠️ No text provided."
            )
        
        # Use provided reference files or stored ones
        refs = reference_files or self._reference_files
        if not refs:
            return SynthesisResult(
                success=False,
                audio_path="",
                message="⚠️ No reference audio files set. Please upload voice samples first."
            )
        
        # Validate language
        if language not in self.SUPPORTED_LANGUAGES:
            return SynthesisResult(
                success=False,
                audio_path="",
                message=f"⚠️ Unsupported language: {language}. Supported: {', '.join(self.SUPPORTED_LANGUAGES)}"
            )
        
        # Generate output path if not provided
        if output_path is None:
            import time
            timestamp = int(time.time() * 1000)
            output_path = str(self.output_dir / f"output_{timestamp}.wav")
        
        try:
            model = self._load_model()
            
            print(f"🗣️ Generating speech: '{text[:50]}...'")
            
            # XTTS generation
            model.tts_to_file(
                text=text,
                speaker_wav=refs,
                language=language,
                file_path=output_path
            )
            
            return SynthesisResult(
                success=True,
                audio_path=output_path,
                message="✅ Audio generated successfully!"
            )
            
        except Exception as e:
            return SynthesisResult(
                success=False,
                audio_path="",
                message=f"❌ Synthesis failed: {str(e)}"
            )
    
    def speak(
        self,
        text: str,
        profile_audio_dir: str,
        language: str = "en",
        output_path: Optional[str] = None
    ) -> SynthesisResult:
        """
        High-level method: Find reference audio in a directory and synthesize.
        
        Args:
            text: Text to synthesize
            profile_audio_dir: Directory containing reference audio files
            language: Language code
            output_path: Output file path
            
        Returns:
            SynthesisResult
        """
        # Find audio files in the directory
        audio_dir = Path(profile_audio_dir)
        if not audio_dir.exists():
            return SynthesisResult(
                success=False,
                audio_path="",
                message=f"⚠️ Audio directory not found: {profile_audio_dir}"
            )
        
        extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
        audio_files = [
            str(f) for f in audio_dir.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]
        
        if not audio_files:
            return SynthesisResult(
                success=False,
                audio_path="",
                message="⚠️ No audio files found in the profile. Upload voice samples first."
            )
        
        # Preprocess to WAV if needed
        wav_files = []
        for f in audio_files:
            if f.endswith(".wav"):
                wav_files.append(f)
            else:
                # Convert to WAV
                result = self.preprocess_audio([f], str(audio_dir / "processed"))
                if result.success:
                    wav_files.extend(result.processed_files)
        
        if not wav_files:
            return SynthesisResult(
                success=False,
                audio_path="",
                message="⚠️ Failed to prepare reference audio files."
            )
        
        # Synthesize
        return self.synthesize(text, language, wav_files, output_path)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.SUPPORTED_LANGUAGES.copy()


# Singleton instance
_mouth_instance: Optional[Mouth] = None


def get_mouth(
    device: Optional[str] = None,
    output_dir: str = "output"
) -> Mouth:
    """
    Get or create a singleton Mouth instance.
    
    Args:
        device: "cuda" or "cpu"
        output_dir: Directory for output files
        
    Returns:
        Mouth instance
    """
    global _mouth_instance
    if _mouth_instance is None:
        _mouth_instance = Mouth(device=device, output_dir=output_dir)
    return _mouth_instance
