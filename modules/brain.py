"""
Brain Module - RAG with ChromaDB + Llama 3
==========================================
Ingests transcripts into ChromaDB for semantic search.
Uses llama-cpp-python to generate persona-aware responses.

NO GRADIO DEPENDENCIES - Pure data in/out.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class VectorizeResult:
    """Result of vectorization operation."""
    success: bool
    message: str
    chunks_count: int = 0


@dataclass
class SearchResult:
    """Result from semantic search."""
    texts: List[str] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)


@dataclass
class GenerationResult:
    """Result of text generation."""
    success: bool
    response: str
    message: str
    context_used: List[str] = field(default_factory=list)


class Brain:
    """
    Brain module for RAG-powered persona responses.
    
    Uses:
    - ChromaDB for vector storage and semantic search
    - Llama 3 (via llama-cpp-python) for text generation
    
    The flow:
    1. Ingest transcripts into ChromaDB
    2. On query: search for relevant context
    3. Inject context into system prompt
    4. Generate response with Llama 3
    """
    
    # Default system prompt template
    DEFAULT_SYSTEM_PROMPT = """You are embodying the persona of a comedian/character. Your responses should match their unique speaking style, vocabulary, and humor.

IMPORTANT STYLE GUIDELINES:
- Match the tone, rhythm, and vocabulary from the context provided
- Use similar jokes, callbacks, and comedic timing
- Stay in character at all times
- Be natural and conversational

Here are examples of their speaking style (use these to match their voice):

{context}

---

Now respond to the user in this exact style. Be authentic to the character."""
    
    def __init__(
        self,
        collection_name: str = "persona_context",
        persist_directory: Optional[str] = None,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,  # -1 = use all GPU layers
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the Brain module.
        
        Args:
            collection_name: Name for the ChromaDB collection
            persist_directory: Path for ChromaDB persistence
            model_path: Path to Llama 3 GGUF model file
            n_ctx: Context window size for Llama
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            embedding_model: Sentence transformer model for embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.embedding_model_name = embedding_model
        
        self._chroma_client = None
        self._collection = None
        self._llm = None
        self._embedding_model = None
    
    def _get_embedding_function(self):
        """Get or create the embedding function."""
        if self._embedding_model is None:
            try:
                from chromadb.utils import embedding_functions
                self._embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_model_name
                )
                print(f"✅ Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                print(f"❌ Failed to load embedding model: {e}")
                raise
        return self._embedding_model
    
    def _get_collection(self):
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                if self.persist_directory:
                    # Persistent storage
                    os.makedirs(self.persist_directory, exist_ok=True)
                    self._chroma_client = chromadb.PersistentClient(
                        path=self.persist_directory,
                        settings=Settings(anonymized_telemetry=False)
                    )
                else:
                    # In-memory (for testing)
                    self._chroma_client = chromadb.Client(
                        Settings(anonymized_telemetry=False)
                    )
                
                self._collection = self._chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self._get_embedding_function(),
                    metadata={"hnsw:space": "cosine"}
                )
                print(f"✅ ChromaDB collection '{self.collection_name}' ready")
                
            except Exception as e:
                print(f"❌ Failed to initialize ChromaDB: {e}")
                raise
        
        return self._collection
    
    def _load_llm(self):
        """Load the Llama model."""
        if self._llm is None:
            if not self.model_path:
                raise ValueError("No model path provided. Set model_path to a valid GGUF file.")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            try:
                from llama_cpp import Llama
                
                print(f"⏳ Loading Llama model from: {self.model_path}")
                self._llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False
                )
                print("✅ Llama model loaded successfully!")
                
            except Exception as e:
                print(f"❌ Failed to load Llama model: {e}")
                raise
        
        return self._llm
    
    def _chunk_text(
        self, 
        text: str, 
        chunk_size: int = 500, 
        overlap: int = 50
    ) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Target characters per chunk
            overlap: Character overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        # Split into sentences (rough approximation)
        sentences = []
        current = ""
        for char in text:
            current += char
            if char in ".!?" and len(current) > 20:
                sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())
        
        # Combine sentences into chunks
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If no natural sentence breaks, chunk by size
        if not chunks and text:
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())
        
        return chunks
    
    def ingest_text(
        self,
        text: str,
        source_id: str = "transcript",
        clear_existing: bool = True
    ) -> VectorizeResult:
        """
        Ingest text into the vector store.
        
        Args:
            text: Text to ingest
            source_id: Identifier for the source
            clear_existing: Whether to clear existing documents first
            
        Returns:
            VectorizeResult
        """
        if not text.strip():
            return VectorizeResult(
                success=False,
                message="No text to ingest."
            )
        
        try:
            collection = self._get_collection()
            
            # Clear existing if requested
            if clear_existing:
                try:
                    # Get all IDs and delete
                    existing = collection.get()
                    if existing["ids"]:
                        collection.delete(ids=existing["ids"])
                        print(f"🗑️ Cleared {len(existing['ids'])} existing documents")
                except Exception:
                    pass  # Collection might be empty
            
            # Chunk the text
            chunks = self._chunk_text(text)
            
            if not chunks:
                return VectorizeResult(
                    success=False,
                    message="No chunks generated from text."
                )
            
            # Add to collection
            ids = [f"{source_id}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": source_id, "chunk_index": i} for i in range(len(chunks))]
            
            collection.add(
                ids=ids,
                documents=chunks,
                metadatas=metadatas
            )
            
            return VectorizeResult(
                success=True,
                message=f"✅ Ingested {len(chunks)} chunks into vector store.",
                chunks_count=len(chunks)
            )
            
        except Exception as e:
            return VectorizeResult(
                success=False,
                message=f"❌ Ingestion failed: {str(e)}"
            )
    
    def ingest_file(
        self,
        file_path: str,
        clear_existing: bool = True
    ) -> VectorizeResult:
        """
        Ingest a text file into the vector store.
        
        Args:
            file_path: Path to the text file
            clear_existing: Whether to clear existing documents
            
        Returns:
            VectorizeResult
        """
        if not os.path.exists(file_path):
            return VectorizeResult(
                success=False,
                message=f"File not found: {file_path}"
            )
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            source_id = Path(file_path).stem
            return self.ingest_text(text, source_id, clear_existing)
            
        except Exception as e:
            return VectorizeResult(
                success=False,
                message=f"❌ Failed to read file: {str(e)}"
            )
    
    def search(
        self,
        query: str,
        n_results: int = 5
    ) -> SearchResult:
        """
        Search for relevant context.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            SearchResult with matching texts
        """
        try:
            collection = self._get_collection()
            
            # Check if collection has documents
            count = collection.count()
            if count == 0:
                return SearchResult(texts=[], distances=[])
            
            results = collection.query(
                query_texts=[query],
                n_results=min(n_results, count)
            )
            
            texts = results["documents"][0] if results["documents"] else []
            distances = results["distances"][0] if results.get("distances") else []
            
            return SearchResult(texts=texts, distances=distances)
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return SearchResult(texts=[], distances=[])
    
    def generate(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        n_context_chunks: int = 5,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> GenerationResult:
        """
        Generate a response using RAG.
        
        Args:
            user_input: User's message
            system_prompt: Custom system prompt (uses default if None)
            n_context_chunks: Number of context chunks to retrieve
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            top_p: Top-p sampling
            
        Returns:
            GenerationResult
        """
        if not user_input.strip():
            return GenerationResult(
                success=False,
                response="",
                message="No input provided."
            )
        
        try:
            # Search for relevant context
            search_results = self.search(user_input, n_context_chunks)
            context_texts = search_results.texts
            
            # Build context string
            if context_texts:
                context = "\n\n".join([f"• {t}" for t in context_texts])
            else:
                context = "(No specific examples found - respond naturally while staying in character)"
            
            # Build system prompt
            if system_prompt is None:
                system_prompt = self.DEFAULT_SYSTEM_PROMPT
            
            final_system_prompt = system_prompt.format(context=context)
            
            # Load LLM and generate
            llm = self._load_llm()
            
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": final_system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["User:", "Human:", "\n\n---"]
            )
            
            generated_text = response["choices"][0]["message"]["content"].strip()
            
            return GenerationResult(
                success=True,
                response=generated_text,
                message="✅ Response generated.",
                context_used=context_texts
            )
            
        except Exception as e:
            return GenerationResult(
                success=False,
                response="",
                message=f"❌ Generation failed: {str(e)}"
            )
    
    def get_collection_stats(self) -> dict:
        """
        Get statistics about the current collection.
        
        Returns:
            Dictionary with collection stats
        """
        try:
            collection = self._get_collection()
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            return {
                "error": str(e)
            }
    
    def clear_collection(self) -> Tuple[bool, str]:
        """
        Clear all documents from the collection.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            collection = self._get_collection()
            existing = collection.get()
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
            return True, "✅ Collection cleared."
        except Exception as e:
            return False, f"❌ Failed to clear collection: {str(e)}"


# Singleton instance holder
_brain_instances: dict = {}


def get_brain(
    profile_name: str,
    vector_db_dir: str,
    model_path: Optional[str] = None,
    **kwargs
) -> Brain:
    """
    Get or create a Brain instance for a profile.
    
    Args:
        profile_name: Profile identifier
        vector_db_dir: Path to vector database directory
        model_path: Path to Llama model
        **kwargs: Additional Brain configuration
        
    Returns:
        Brain instance
    """
    global _brain_instances
    
    if profile_name not in _brain_instances:
        _brain_instances[profile_name] = Brain(
            collection_name=f"persona_{profile_name}",
            persist_directory=vector_db_dir,
            model_path=model_path,
            **kwargs
        )
    
    return _brain_instances[profile_name]
