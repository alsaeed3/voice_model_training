# Digital Human Clone Dashboard - Modules Package
"""
This package contains the core functional modules:
- ear.py: Transcription using faster-whisper
- brain.py: RAG with ChromaDB + Llama 3
- mouth.py: Voice synthesis using Coqui XTTS v2
- profile_manager.py: Profile folder management
- orchestrator.py: One-click training pipeline
"""

from .ear import Ear
from .brain import Brain
from .mouth import Mouth
from .profile_manager import ProfileManager
from .orchestrator import Orchestrator

__all__ = ["Ear", "Brain", "Mouth", "ProfileManager", "Orchestrator"]
