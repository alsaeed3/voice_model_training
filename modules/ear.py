"""
Ear Module - Transcription
==========================
Uses faster-whisper to transcribe audio files.
Saves transcripts to the profile's transcripts.txt file.

NO GRADIO DEPENDENCIES - Pure data in/out.
"""

import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""
    success: bool
    text: str
    message: str
    segments_count: int = 0
    duration_seconds: float = 0.0


class Ear:
    """
    Transcription module using faster-whisper.
    
    Supports multiple Whisper model sizes:
    - tiny, base, small, medium, large-v2, large-v3
    """
    
    def __init__(
        self, 
        model_size: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16"
    ):
        """
        Initialize the Ear (transcription) module.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
            device: "cuda" or "cpu"
            compute_type: "float16" for GPU, "int8" or "float32" for CPU
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None
    
    def _load_model(self):
        """Lazy load the whisper model."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                print(f"⏳ Loading Whisper model '{self.model_size}' on {self.device}...")
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type
                )
                print("✅ Whisper model loaded successfully!")
            except Exception as e:
                print(f"❌ Failed to load Whisper model: {e}")
                raise
        return self._model
    
    def transcribe_file(
        self, 
        audio_path: str,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe a single audio file.
        
        Args:
            audio_path: Path to the audio file
            language: Optional language code (e.g., "en", "es"). Auto-detect if None.
            
        Returns:
            TranscriptionResult with the transcribed text
        """
        if not os.path.exists(audio_path):
            return TranscriptionResult(
                success=False,
                text="",
                message=f"Audio file not found: {audio_path}"
            )
        
        try:
            model = self._load_model()
            
            # Transcribe
            segments, info = model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                vad_filter=True,  # Filter out non-speech
                vad_parameters=dict(
                    min_silence_duration_ms=500
                )
            )
            
            # Collect all segments
            text_parts = []
            segment_count = 0
            for segment in segments:
                text_parts.append(segment.text.strip())
                segment_count += 1
            
            full_text = " ".join(text_parts)
            
            return TranscriptionResult(
                success=True,
                text=full_text,
                message=f"✅ Transcribed {segment_count} segments ({info.duration:.1f}s)",
                segments_count=segment_count,
                duration_seconds=info.duration
            )
            
        except Exception as e:
            return TranscriptionResult(
                success=False,
                text="",
                message=f"❌ Transcription failed: {str(e)}"
            )
    
    def transcribe_multiple(
        self,
        audio_paths: List[str],
        language: Optional[str] = None,
        separator: str = "\n\n---\n\n"
    ) -> TranscriptionResult:
        """
        Transcribe multiple audio files and combine the results.
        
        Args:
            audio_paths: List of audio file paths
            language: Optional language code
            separator: Separator between transcriptions
            
        Returns:
            Combined TranscriptionResult
        """
        if not audio_paths:
            return TranscriptionResult(
                success=False,
                text="",
                message="No audio files provided."
            )
        
        all_texts = []
        total_segments = 0
        total_duration = 0.0
        errors = []
        
        for idx, audio_path in enumerate(audio_paths):
            print(f"🎧 Transcribing [{idx + 1}/{len(audio_paths)}]: {Path(audio_path).name}")
            result = self.transcribe_file(audio_path, language)
            
            if result.success:
                all_texts.append(result.text)
                total_segments += result.segments_count
                total_duration += result.duration_seconds
            else:
                errors.append(f"{Path(audio_path).name}: {result.message}")
        
        if not all_texts:
            return TranscriptionResult(
                success=False,
                text="",
                message=f"All transcriptions failed. Errors: {'; '.join(errors)}"
            )
        
        combined_text = separator.join(all_texts)
        
        success_count = len(all_texts)
        total_count = len(audio_paths)
        
        message = f"✅ Transcribed {success_count}/{total_count} files ({total_segments} segments, {total_duration:.1f}s total)"
        if errors:
            message += f"\n⚠️ Errors: {'; '.join(errors)}"
        
        return TranscriptionResult(
            success=True,
            text=combined_text,
            message=message,
            segments_count=total_segments,
            duration_seconds=total_duration
        )
    
    def transcribe_and_save(
        self,
        audio_paths: List[str],
        output_path: str,
        language: Optional[str] = None,
        append: bool = False
    ) -> TranscriptionResult:
        """
        Transcribe multiple audio files and save to a text file.
        
        Args:
            audio_paths: List of audio file paths
            output_path: Path to save the transcription
            language: Optional language code
            append: If True, append to existing file
            
        Returns:
            TranscriptionResult
        """
        result = self.transcribe_multiple(audio_paths, language)
        
        if not result.success:
            return result
        
        try:
            mode = "a" if append else "w"
            with open(output_path, mode, encoding="utf-8") as f:
                if append:
                    f.write("\n\n---\n\n")
                f.write(result.text)
            
            result.message += f"\n📄 Saved to: {output_path}"
            return result
            
        except Exception as e:
            return TranscriptionResult(
                success=False,
                text=result.text,
                message=f"❌ Failed to save transcript: {str(e)}"
            )


# Singleton instance for reuse
_ear_instance: Optional[Ear] = None


def get_ear(
    model_size: str = "medium",
    device: str = "cuda",
    compute_type: str = "float16"
) -> Ear:
    """
    Get or create a singleton Ear instance.
    
    Args:
        model_size: Whisper model size
        device: "cuda" or "cpu"
        compute_type: Compute type for the model
        
    Returns:
        Ear instance
    """
    global _ear_instance
    if _ear_instance is None:
        _ear_instance = Ear(model_size, device, compute_type)
    return _ear_instance
