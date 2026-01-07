"""
Text chunker for RAG-ready output.
"""
from typing import List, Optional

from resarchflow.core.schemas import Chunk, Section, SectionType


class TextChunker:
    """
    Chunks text for RAG systems.
    Supports multiple strategies.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in tokens (approx words)
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_text(
        self,
        text: str,
        section_type: SectionType = SectionType.FULL_TEXT
    ) -> List[Chunk]:
        """
        Chunk text into RAG-ready pieces.
        
        Args:
            text: Text to chunk
            section_type: Section type for metadata
            
        Returns:
            List of Chunk objects
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            return []
        
        # Split into paragraphs first
        paragraphs = self._split_paragraphs(text)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for para in paragraphs:
            # If adding paragraph exceeds chunk size
            if len(current_chunk) + len(para) > self.chunk_size * 4:  # ~4 chars per word
                if current_chunk:
                    chunks.append(Chunk(
                        section_type=section_type,
                        content=current_chunk.strip(),
                        index=chunk_index,
                        start_char=current_start,
                        end_char=current_start + len(current_chunk)
                    ))
                    chunk_index += 1
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap(current_chunk)
                    current_start = current_start + len(current_chunk) - len(overlap_text)
                    current_chunk = overlap_text
                
            current_chunk += para + "\n\n"
        
        # Add final chunk
        if current_chunk and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(Chunk(
                section_type=section_type,
                content=current_chunk.strip(),
                index=chunk_index,
                start_char=current_start,
                end_char=current_start + len(current_chunk)
            ))
        
        return chunks
    
    def chunk_sections(
        self,
        sections: dict[SectionType, Section]
    ) -> List[Chunk]:
        """
        Chunk all sections.
        
        Args:
            sections: Dict of section type to Section
            
        Returns:
            List of all chunks across sections
        """
        all_chunks = []
        
        # Process sections in order (skip full_text if other sections exist)
        has_other_sections = any(
            st != SectionType.FULL_TEXT for st in sections.keys()
        )
        
        for section_type, section in sections.items():
            # Skip full_text if we have structured sections
            if section_type == SectionType.FULL_TEXT and has_other_sections:
                continue
            
            chunks = self.chunk_text(section.content, section_type)
            all_chunks.extend(chunks)
        
        # Reindex all chunks
        for i, chunk in enumerate(all_chunks):
            chunk.index = i
        
        return all_chunks
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines
        paragraphs = text.split("\n\n")
        
        # Clean and filter
        result = []
        for para in paragraphs:
            para = para.strip()
            if para and len(para) > 20:
                result.append(para)
        
        return result
    
    def _get_overlap(self, text: str) -> str:
        """Get overlap text from end of chunk."""
        words = text.split()
        overlap_words = words[-self.chunk_overlap:] if len(words) > self.chunk_overlap else words
        return " ".join(overlap_words)


class SemanticChunker:
    """
    Semantic-aware chunker that tries to preserve meaning.
    Uses sentence boundaries and semantic similarity.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._sentence_splitter = None
    
    def _get_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(
        self,
        text: str,
        section_type: SectionType = SectionType.FULL_TEXT
    ) -> List[Chunk]:
        """Chunk text preserving sentence boundaries."""
        sentences = self._get_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(Chunk(
                        section_type=section_type,
                        content=" ".join(current_chunk),
                        index=chunk_index
                    ))
                    chunk_index += 1
                    
                    # Keep last few sentences for overlap
                    overlap_count = min(2, len(current_chunk))
                    current_chunk = current_chunk[-overlap_count:]
                    current_length = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunks.append(Chunk(
                section_type=section_type,
                content=" ".join(current_chunk),
                index=chunk_index
            ))
        
        return chunks
