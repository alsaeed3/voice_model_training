"""
Profile Manager Module
======================
Handles creation, loading, and management of profile folders.
Each profile is a self-contained folder with audio, transcripts, and vector DB.

NO GRADIO DEPENDENCIES - Pure data in/out.
"""

import os
import json
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class ProfileMetadata(BaseModel):
    """Metadata for a profile."""
    name: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    audio_count: int = 0
    is_transcribed: bool = False
    is_vectorized: bool = False


class ChatMessage(BaseModel):
    """Single chat message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    audio_path: Optional[str] = None


class ChatHistory(BaseModel):
    """Chat history for a profile."""
    messages: List[ChatMessage] = Field(default_factory=list)


class ProfileManager:
    """
    Manages profile folders and their contents.
    
    Profile Structure:
        data/profiles/{profile_name}/
        ├── audio/              # Uploaded MP3s
        ├── vector_db/          # ChromaDB persistence
        ├── transcripts.txt     # Raw transcription text
        ├── chat_history.json   # Conversation history
        └── metadata.json       # Profile metadata
    """
    
    BASE_DIR = Path("data/profiles")
    
    def __init__(self):
        """Initialize the profile manager."""
        self.BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    def _get_profile_path(self, profile_name: str) -> Path:
        """Get the absolute path for a profile."""
        # Sanitize profile name
        safe_name = "".join(c for c in profile_name if c.isalnum() or c in ("-", "_", " "))
        safe_name = safe_name.strip().replace(" ", "_")
        return self.BASE_DIR / safe_name
    
    def create_profile(self, profile_name: str, description: str = "") -> tuple[bool, str]:
        """
        Create a new profile with all required subdirectories.
        
        Args:
            profile_name: Name for the new profile
            description: Optional description
            
        Returns:
            Tuple of (success, message)
        """
        if not profile_name or not profile_name.strip():
            return False, "Profile name cannot be empty."
        
        profile_path = self._get_profile_path(profile_name)
        
        if profile_path.exists():
            return False, f"Profile '{profile_name}' already exists."
        
        try:
            # Create directory structure
            (profile_path / "audio").mkdir(parents=True, exist_ok=True)
            (profile_path / "vector_db").mkdir(parents=True, exist_ok=True)
            
            # Initialize empty transcripts
            (profile_path / "transcripts.txt").touch()
            
            # Initialize metadata
            metadata = ProfileMetadata(
                name=profile_name.strip(),
                description=description
            )
            with open(profile_path / "metadata.json", "w") as f:
                f.write(metadata.model_dump_json(indent=2))
            
            # Initialize empty chat history
            chat_history = ChatHistory()
            with open(profile_path / "chat_history.json", "w") as f:
                f.write(chat_history.model_dump_json(indent=2))
            
            return True, f"✅ Profile '{profile_name}' created successfully!"
            
        except Exception as e:
            return False, f"❌ Failed to create profile: {str(e)}"
    
    def list_profiles(self) -> List[str]:
        """
        List all available profile names.
        
        Returns:
            List of profile folder names
        """
        if not self.BASE_DIR.exists():
            return []
        
        profiles = []
        for item in self.BASE_DIR.iterdir():
            if item.is_dir() and (item / "metadata.json").exists():
                profiles.append(item.name)
        
        return sorted(profiles)
    
    def get_profile_metadata(self, profile_name: str) -> Optional[ProfileMetadata]:
        """
        Get metadata for a profile.
        
        Args:
            profile_name: Name of the profile
            
        Returns:
            ProfileMetadata or None if not found
        """
        profile_path = self._get_profile_path(profile_name)
        metadata_path = profile_path / "metadata.json"
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, "r") as f:
                data = json.load(f)
            return ProfileMetadata(**data)
        except Exception:
            return None
    
    def update_metadata(self, profile_name: str, **updates) -> bool:
        """
        Update profile metadata.
        
        Args:
            profile_name: Name of the profile
            **updates: Fields to update
            
        Returns:
            Success status
        """
        profile_path = self._get_profile_path(profile_name)
        metadata_path = profile_path / "metadata.json"
        
        if not metadata_path.exists():
            return False
        
        try:
            metadata = self.get_profile_metadata(profile_name)
            if not metadata:
                return False
            
            # Update fields
            metadata_dict = metadata.model_dump()
            metadata_dict.update(updates)
            updated_metadata = ProfileMetadata(**metadata_dict)
            
            with open(metadata_path, "w") as f:
                f.write(updated_metadata.model_dump_json(indent=2))
            
            return True
        except Exception:
            return False
    
    def get_audio_dir(self, profile_name: str) -> Path:
        """Get the audio directory path for a profile."""
        return self._get_profile_path(profile_name) / "audio"
    
    def get_vector_db_dir(self, profile_name: str) -> Path:
        """Get the vector DB directory path for a profile."""
        return self._get_profile_path(profile_name) / "vector_db"
    
    def get_transcripts_path(self, profile_name: str) -> Path:
        """Get the transcripts file path for a profile."""
        return self._get_profile_path(profile_name) / "transcripts.txt"
    
    def get_chat_history_path(self, profile_name: str) -> Path:
        """Get the chat history file path for a profile."""
        return self._get_profile_path(profile_name) / "chat_history.json"
    
    def save_uploaded_audio(self, profile_name: str, file_path: str) -> tuple[bool, str]:
        """
        Copy an uploaded audio file to the profile's audio directory.
        
        Args:
            profile_name: Target profile name
            file_path: Path to the uploaded file
            
        Returns:
            Tuple of (success, saved_path or error_message)
        """
        import shutil
        
        audio_dir = self.get_audio_dir(profile_name)
        if not audio_dir.exists():
            return False, f"Profile '{profile_name}' not found."
        
        try:
            source = Path(file_path)
            dest = audio_dir / source.name
            
            # Handle duplicate names
            counter = 1
            while dest.exists():
                stem = source.stem
                suffix = source.suffix
                dest = audio_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            shutil.copy2(file_path, dest)
            
            # Update audio count in metadata
            audio_count = len(list(audio_dir.glob("*")))
            self.update_metadata(profile_name, audio_count=audio_count)
            
            return True, str(dest)
            
        except Exception as e:
            return False, f"Failed to save audio: {str(e)}"
    
    def get_audio_files(self, profile_name: str) -> List[Path]:
        """
        Get all audio files in a profile.
        
        Args:
            profile_name: Profile name
            
        Returns:
            List of audio file paths
        """
        audio_dir = self.get_audio_dir(profile_name)
        if not audio_dir.exists():
            return []
        
        extensions = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
        files = []
        for f in audio_dir.iterdir():
            if f.is_file() and f.suffix.lower() in extensions:
                files.append(f)
        
        return sorted(files)
    
    def load_chat_history(self, profile_name: str) -> ChatHistory:
        """
        Load chat history for a profile.
        
        Args:
            profile_name: Profile name
            
        Returns:
            ChatHistory object
        """
        chat_path = self.get_chat_history_path(profile_name)
        
        if not chat_path.exists():
            return ChatHistory()
        
        try:
            with open(chat_path, "r") as f:
                data = json.load(f)
            return ChatHistory(**data)
        except Exception:
            return ChatHistory()
    
    def save_chat_message(
        self, 
        profile_name: str, 
        role: str, 
        content: str, 
        audio_path: Optional[str] = None
    ) -> bool:
        """
        Append a message to chat history.
        
        Args:
            profile_name: Profile name
            role: "user" or "assistant"
            content: Message text
            audio_path: Optional path to generated audio
            
        Returns:
            Success status
        """
        chat_path = self.get_chat_history_path(profile_name)
        
        try:
            history = self.load_chat_history(profile_name)
            message = ChatMessage(
                role=role,
                content=content,
                audio_path=audio_path
            )
            history.messages.append(message)
            
            with open(chat_path, "w") as f:
                f.write(history.model_dump_json(indent=2))
            
            return True
        except Exception:
            return False
    
    def clear_chat_history(self, profile_name: str) -> bool:
        """
        Clear chat history for a profile.
        
        Args:
            profile_name: Profile name
            
        Returns:
            Success status
        """
        chat_path = self.get_chat_history_path(profile_name)
        
        try:
            empty_history = ChatHistory()
            with open(chat_path, "w") as f:
                f.write(empty_history.model_dump_json(indent=2))
            return True
        except Exception:
            return False
    
    def profile_exists(self, profile_name: str) -> bool:
        """Check if a profile exists."""
        profile_path = self._get_profile_path(profile_name)
        return profile_path.exists() and (profile_path / "metadata.json").exists()
    
    def delete_profile(self, profile_name: str) -> tuple[bool, str]:
        """
        Delete a profile and all its contents.
        
        Args:
            profile_name: Profile to delete
            
        Returns:
            Tuple of (success, message)
        """
        import shutil
        
        profile_path = self._get_profile_path(profile_name)
        
        if not profile_path.exists():
            return False, f"Profile '{profile_name}' not found."
        
        try:
            shutil.rmtree(profile_path)
            return True, f"✅ Profile '{profile_name}' deleted."
        except Exception as e:
            return False, f"❌ Failed to delete profile: {str(e)}"
