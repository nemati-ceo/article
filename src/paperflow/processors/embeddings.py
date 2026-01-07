"""
Embedding processor for vector storage.
"""
from typing import Any, List, Optional

from paperflow.schemas import Chunk, Paper


class EmbeddingProcessor:
    """Generates embeddings for paper chunks."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        backend: str = "auto"
    ):
        self.model_name = model_name
        self.backend = backend
        self._model = None
        self._embed_func = None
        self.available = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize embedding model."""
        if self.backend == "auto":
            self._try_sentence_transformers()
            if not self.available:
                self._try_openai()
        elif self.backend == "sentence_transformers":
            self._try_sentence_transformers()
        elif self.backend == "openai":
            self._try_openai()

    def _try_sentence_transformers(self) -> None:
        """Try to load sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._embed_func = self._embed_st
            self.available = True
            self.backend = "sentence_transformers"
        except ImportError:
            pass

    def _try_openai(self) -> None:
        """Try to use OpenAI embeddings."""
        try:
            import os
            if os.getenv("OPENAI_API_KEY"):
                self._embed_func = self._embed_openai
                self.available = True
                self.backend = "openai"
        except ImportError:
            pass

    def _embed_st(self, texts: List[str]) -> List[List[float]]:
        """Embed using sentence-transformers."""
        embeddings = self._model.encode(texts)
        return embeddings.tolist()

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """Embed using OpenAI API."""
        import openai
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [e.embedding for e in response.data]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        if not self.available:
            raise RuntimeError(
                "No embedding backend available. "
                "Install sentence-transformers or set OPENAI_API_KEY"
            )
        return self._embed_func(texts)

    def embed_chunks(
        self,
        chunks: List[Chunk],
        batch_size: int = 32
    ) -> List[tuple[Chunk, List[float]]]:
        """Generate embeddings for chunks."""
        results = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.content for c in batch]
            embeddings = self.embed_texts(texts)

            for chunk, emb in zip(batch, embeddings):
                results.append((chunk, emb))

        return results

    def embed_paper(self, paper: Paper) -> List[tuple[Chunk, List[float]]]:
        """Generate embeddings for all chunks in a paper."""
        if not paper.chunks:
            return []
        return self.embed_chunks(paper.chunks)


class VectorStoreAdapter:
    """Adapter for various vector stores."""

    def __init__(
        self,
        backend: str = "chroma",
        collection_name: str = "papers",
        **kwargs
    ):
        self.backend = backend
        self.collection_name = collection_name
        self._store = None
        self._initialize(**kwargs)

    def _initialize(self, **kwargs) -> None:
        """Initialize vector store backend."""
        if self.backend == "chroma":
            self._init_chroma(**kwargs)
        elif self.backend == "memory":
            self._init_memory()

    def _init_chroma(self, **kwargs) -> None:
        """Initialize ChromaDB."""
        try:
            import chromadb

            persist_dir = kwargs.get("persist_directory")
            if persist_dir:
                client = chromadb.PersistentClient(path=persist_dir)
            else:
                client = chromadb.Client()

            self._store = client.get_or_create_collection(name=self.collection_name)

        except ImportError:
            print("ChromaDB not available. Install with: pip install chromadb")
            self._init_memory()

    def _init_memory(self) -> None:
        """Initialize in-memory store."""
        self._store = {
            "ids": [],
            "embeddings": [],
            "documents": [],
            "metadatas": []
        }
        self.backend = "memory"

    def add(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        paper_uuid: str
    ) -> None:
        """Add chunks with embeddings to store."""
        if self.backend == "chroma":
            self._store.add(
                ids=[f"{paper_uuid}_{c.chunk_id}" for c in chunks],
                embeddings=embeddings,
                documents=[c.content for c in chunks],
                metadatas=[{
                    "paper_uuid": paper_uuid,
                    "section": c.section_type.value,
                    "chunk_index": c.index
                } for c in chunks]
            )
        else:
            for chunk, emb in zip(chunks, embeddings):
                self._store["ids"].append(f"{paper_uuid}_{chunk.chunk_id}")
                self._store["embeddings"].append(emb)
                self._store["documents"].append(chunk.content)
                self._store["metadatas"].append({
                    "paper_uuid": paper_uuid,
                    "section": chunk.section_type.value
                })

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filter_paper: Optional[str] = None
    ) -> List[dict]:
        """Search for similar chunks."""
        if self.backend == "chroma":
            where = {"paper_uuid": filter_paper} if filter_paper else None
            results = self._store.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )

            return [
                {"content": doc, "metadata": meta, "id": id_}
                for doc, meta, id_ in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["ids"][0]
                )
            ]
        else:
            import numpy as np

            query = np.array(query_embedding)
            scores = []

            for i, emb in enumerate(self._store["embeddings"]):
                emb = np.array(emb)
                score = np.dot(query, emb) / (np.linalg.norm(query) * np.linalg.norm(emb))

                if filter_paper:
                    if self._store["metadatas"][i].get("paper_uuid") != filter_paper:
                        continue

                scores.append((score, i))

            scores.sort(reverse=True)

            return [
                {
                    "content": self._store["documents"][i],
                    "metadata": self._store["metadatas"][i],
                    "id": self._store["ids"][i]
                }
                for _, i in scores[:n_results]
            ]
