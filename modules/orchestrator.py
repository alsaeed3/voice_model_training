"""
Orchestrator Module - One-Click Training Pipeline
==================================================
Handles the complete training flow:
1. Ingest: Save uploaded MP3s to profile
2. Ear: Run Whisper transcription
3. Brain: Index transcripts in ChromaDB
4. Mouth: Preprocess audio for TTS

Features:
- Sequential execution with error handling
- Progress callback for UI updates
- Stops pipeline on any failure

NO GRADIO DEPENDENCIES - Pure data in/out.
"""

import os
from pathlib import Path
from typing import List, Optional, Callable, Generator
from dataclasses import dataclass, field
from enum import Enum
import shutil


class TrainingStep(Enum):
    """Training pipeline steps."""
    INGEST = "ingest"
    TRANSCRIBE = "transcribe"
    VECTORIZE = "vectorize"
    PREPROCESS = "preprocess"


@dataclass
class StepResult:
    """Result of a single training step."""
    step: TrainingStep
    success: bool
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class PipelineProgress:
    """Progress update for UI."""
    current_step: int
    total_steps: int
    step_name: str
    step_description: str
    is_complete: bool = False
    is_error: bool = False
    error_message: Optional[str] = None


@dataclass
class PipelineResult:
    """Final result of the training pipeline."""
    success: bool
    message: str
    steps_completed: List[StepResult] = field(default_factory=list)
    audio_count: int = 0
    transcript_chunks: int = 0
    vector_chunks: int = 0


class Orchestrator:
    """
    Orchestrates the complete training pipeline.
    
    This class coordinates between Ear, Brain, and Mouth modules
    to provide a seamless one-click training experience.
    """
    
    # Step descriptions for UI feedback
    STEP_DESCRIPTIONS = {
        TrainingStep.INGEST: "Saving uploaded audio files...",
        TrainingStep.TRANSCRIBE: "Converting speech to text with Whisper AI...",
        TrainingStep.VECTORIZE: "Indexing transcripts for semantic search...",
        TrainingStep.PREPROCESS: "Preparing audio files for voice cloning..."
    }
    
    def __init__(
        self,
        profile_manager,
        ear_instance=None,
        brain_factory=None,
        mouth_instance=None
    ):
        """
        Initialize the Orchestrator.
        
        Args:
            profile_manager: ProfileManager instance
            ear_instance: Optional Ear instance (lazy loaded if None)
            brain_factory: Callable that returns Brain for a profile
            mouth_instance: Optional Mouth instance (lazy loaded if None)
        """
        self.profile_manager = profile_manager
        self._ear = ear_instance
        self._brain_factory = brain_factory
        self._mouth = mouth_instance
    
    def _get_ear(self):
        """Get or create Ear instance."""
        if self._ear is None:
            from modules.ear import Ear
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "float32"
            self._ear = Ear(
                model_size=os.getenv("WHISPER_MODEL_SIZE", "medium"),
                device=device,
                compute_type=compute_type
            )
        return self._ear
    
    def _get_brain(self, profile_name: str):
        """Get Brain instance for a profile."""
        if self._brain_factory:
            return self._brain_factory(profile_name)
        
        # Default brain creation
        from modules.brain import Brain
        vector_db_dir = str(self.profile_manager.get_vector_db_dir(profile_name))
        model_path = os.getenv(
            "LLAMA_MODEL_PATH",
            "models/Llama-3-8B-Instruct-v0.1.Q5_K_M.gguf"
        )
        return Brain(
            collection_name=f"persona_{profile_name}",
            persist_directory=vector_db_dir,
            model_path=model_path if os.path.exists(model_path) else None,
            n_ctx=4096,
            n_gpu_layers=-1
        )
    
    def _get_mouth(self):
        """Get or create Mouth instance."""
        if self._mouth is None:
            from modules.mouth import Mouth
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._mouth = Mouth(device=device, output_dir="output")
        return self._mouth
    
    def _step_ingest(
        self,
        profile_name: str,
        audio_files: List[str]
    ) -> StepResult:
        """
        Step 1: Save uploaded audio files to profile.
        
        Args:
            profile_name: Target profile name
            audio_files: List of file paths to save
            
        Returns:
            StepResult
        """
        if not audio_files:
            return StepResult(
                step=TrainingStep.INGEST,
                success=False,
                message="No audio files provided."
            )
        
        saved_count = 0
        errors = []
        
        for file_path in audio_files:
            success, result = self.profile_manager.save_uploaded_audio(
                profile_name, file_path
            )
            if success:
                saved_count += 1
            else:
                errors.append(result)
        
        if saved_count == 0:
            return StepResult(
                step=TrainingStep.INGEST,
                success=False,
                message=f"Failed to save any files. Errors: {'; '.join(errors)}"
            )
        
        message = f"✅ Saved {saved_count}/{len(audio_files)} audio files."
        if errors:
            message += f" ⚠️ {len(errors)} errors."
        
        return StepResult(
            step=TrainingStep.INGEST,
            success=True,
            message=message,
            details={"saved_count": saved_count, "errors": errors}
        )
    
    def _step_transcribe(self, profile_name: str) -> StepResult:
        """
        Step 2: Transcribe audio files with Whisper.
        
        Args:
            profile_name: Profile to transcribe
            
        Returns:
            StepResult
        """
        audio_files = self.profile_manager.get_audio_files(profile_name)
        
        if not audio_files:
            return StepResult(
                step=TrainingStep.TRANSCRIBE,
                success=False,
                message="No audio files found in profile."
            )
        
        try:
            ear = self._get_ear()
            transcripts_path = str(
                self.profile_manager.get_transcripts_path(profile_name)
            )
            
            result = ear.transcribe_and_save(
                audio_paths=[str(f) for f in audio_files],
                output_path=transcripts_path,
                append=False
            )
            
            if result.success:
                self.profile_manager.update_metadata(
                    profile_name, is_transcribed=True
                )
                return StepResult(
                    step=TrainingStep.TRANSCRIBE,
                    success=True,
                    message=result.message,
                    details={
                        "segments": result.segments_count,
                        "duration": result.duration_seconds
                    }
                )
            else:
                return StepResult(
                    step=TrainingStep.TRANSCRIBE,
                    success=False,
                    message=result.message
                )
                
        except Exception as e:
            return StepResult(
                step=TrainingStep.TRANSCRIBE,
                success=False,
                message=f"❌ Transcription failed: {str(e)}"
            )
    
    def _step_vectorize(self, profile_name: str) -> StepResult:
        """
        Step 3: Index transcripts in ChromaDB.
        
        Args:
            profile_name: Profile to vectorize
            
        Returns:
            StepResult
        """
        transcripts_path = self.profile_manager.get_transcripts_path(profile_name)
        
        if not transcripts_path.exists() or transcripts_path.stat().st_size == 0:
            return StepResult(
                step=TrainingStep.VECTORIZE,
                success=False,
                message="No transcripts available. Transcription may have failed."
            )
        
        try:
            brain = self._get_brain(profile_name)
            result = brain.ingest_file(str(transcripts_path), clear_existing=True)
            
            if result.success:
                self.profile_manager.update_metadata(
                    profile_name, is_vectorized=True
                )
                return StepResult(
                    step=TrainingStep.VECTORIZE,
                    success=True,
                    message=result.message,
                    details={"chunks": result.chunks_count}
                )
            else:
                return StepResult(
                    step=TrainingStep.VECTORIZE,
                    success=False,
                    message=result.message
                )
                
        except Exception as e:
            return StepResult(
                step=TrainingStep.VECTORIZE,
                success=False,
                message=f"❌ Vectorization failed: {str(e)}"
            )
    
    def _step_preprocess(self, profile_name: str) -> StepResult:
        """
        Step 4: Preprocess audio for TTS voice cloning.
        
        Converts all audio files to WAV format suitable for XTTS.
        
        Args:
            profile_name: Profile to preprocess
            
        Returns:
            StepResult
        """
        audio_dir = self.profile_manager.get_audio_dir(profile_name)
        audio_files = self.profile_manager.get_audio_files(profile_name)
        
        if not audio_files:
            return StepResult(
                step=TrainingStep.PREPROCESS,
                success=False,
                message="No audio files to preprocess."
            )
        
        try:
            mouth = self._get_mouth()
            processed_dir = audio_dir / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            result = mouth.preprocess_audio(
                file_paths=[str(f) for f in audio_files],
                output_dir=str(processed_dir)
            )
            
            return StepResult(
                step=TrainingStep.PREPROCESS,
                success=result.success,
                message=result.message,
                details={"processed_files": len(result.processed_files)}
            )
            
        except Exception as e:
            return StepResult(
                step=TrainingStep.PREPROCESS,
                success=False,
                message=f"❌ Audio preprocessing failed: {str(e)}"
            )
    
    def run_training_pipeline(
        self,
        profile_name: str,
        audio_files: Optional[List[str]] = None,
        skip_ingest: bool = False
    ) -> Generator[PipelineProgress, None, PipelineResult]:
        """
        Run the complete training pipeline with progress updates.
        
        This is a generator that yields progress updates for each step.
        The final return value is the complete PipelineResult.
        
        Usage:
            pipeline = orchestrator.run_training_pipeline(profile, files)
            for progress in pipeline:
                update_ui(progress)
            result = pipeline.value  # or handle StopIteration
        
        Args:
            profile_name: Target profile name
            audio_files: Optional list of new audio files to add
            skip_ingest: If True, skip the ingest step (use existing files)
            
        Yields:
            PipelineProgress updates
            
        Returns:
            PipelineResult with final status
        """
        steps = [
            (TrainingStep.INGEST, self._step_ingest),
            (TrainingStep.TRANSCRIBE, self._step_transcribe),
            (TrainingStep.VECTORIZE, self._step_vectorize),
            (TrainingStep.PREPROCESS, self._step_preprocess),
        ]
        
        # Skip ingest if requested or no new files
        if skip_ingest or not audio_files:
            steps = steps[1:]  # Remove ingest step
        
        total_steps = len(steps)
        completed_steps: List[StepResult] = []
        
        for idx, (step_type, step_func) in enumerate(steps, start=1):
            # Yield progress before starting step
            yield PipelineProgress(
                current_step=idx,
                total_steps=total_steps,
                step_name=step_type.value.title(),
                step_description=self.STEP_DESCRIPTIONS[step_type]
            )
            
            # Execute step
            if step_type == TrainingStep.INGEST:
                result = step_func(profile_name, audio_files)
            else:
                result = step_func(profile_name)
            
            completed_steps.append(result)
            
            # Check for failure - stop pipeline
            if not result.success:
                yield PipelineProgress(
                    current_step=idx,
                    total_steps=total_steps,
                    step_name=step_type.value.title(),
                    step_description=result.message,
                    is_error=True,
                    error_message=result.message
                )
                
                return PipelineResult(
                    success=False,
                    message=f"Pipeline failed at step {idx}/{total_steps}: {step_type.value}",
                    steps_completed=completed_steps
                )
        
        # All steps completed successfully
        yield PipelineProgress(
            current_step=total_steps,
            total_steps=total_steps,
            step_name="Complete",
            step_description="✅ Training pipeline completed successfully!",
            is_complete=True
        )
        
        # Gather statistics
        audio_count = len(self.profile_manager.get_audio_files(profile_name))
        vector_chunks = 0
        
        for step in completed_steps:
            if step.step == TrainingStep.VECTORIZE:
                vector_chunks = step.details.get("chunks", 0)
        
        return PipelineResult(
            success=True,
            message="✅ Training complete! Your digital human is ready.",
            steps_completed=completed_steps,
            audio_count=audio_count,
            vector_chunks=vector_chunks
        )
    
    def run_training_sync(
        self,
        profile_name: str,
        audio_files: Optional[List[str]] = None,
        skip_ingest: bool = False,
        progress_callback: Optional[Callable[[PipelineProgress], None]] = None
    ) -> PipelineResult:
        """
        Synchronous version of run_training_pipeline.
        
        Args:
            profile_name: Target profile name
            audio_files: Optional audio files to add
            skip_ingest: Skip the ingest step
            progress_callback: Optional callback for progress updates
            
        Returns:
            PipelineResult
        """
        pipeline = self.run_training_pipeline(
            profile_name, audio_files, skip_ingest
        )
        
        result = None
        try:
            while True:
                progress = next(pipeline)
                if progress_callback:
                    progress_callback(progress)
        except StopIteration as e:
            result = e.value
        
        return result or PipelineResult(
            success=False,
            message="Pipeline returned no result."
        )


# Singleton instance
_orchestrator_instance: Optional[Orchestrator] = None


def get_orchestrator(profile_manager, **kwargs) -> Orchestrator:
    """
    Get or create a singleton Orchestrator instance.
    
    Args:
        profile_manager: ProfileManager instance
        **kwargs: Additional Orchestrator configuration
        
    Returns:
        Orchestrator instance
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = Orchestrator(profile_manager, **kwargs)
    return _orchestrator_instance
