"""
Main pipeline orchestrator for paperflow.
"""
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from paperflow.schemas import (
    Citation,
    Paper,
    PaperMetadata,
    ProcessingStatus,
    SearchResult,
    Section,
    SectionType,
    SourceType,
)
from paperflow.providers import UnifiedSearch, get_provider
from paperflow.processors import (
    EmbeddingProcessor,
    MarkerProcessor,
    PDFExtractor,
    TextChunker,
    VectorStoreAdapter,
)


class PaperPipeline:
    """
    Main pipeline for paper search, download, extraction, and RAG.
    
    Args:
        pdf_dir: Directory to store downloaded PDFs
        markdown_dir: Directory to store extracted markdown
        db_path: Path to vector database
        vector_store: Vector store backend ("chroma", "faiss", etc.)
        embedding_model: Sentence transformer model name
        gpu: Enable GPU acceleration for extraction and embeddings
        extraction_backend: PDF extraction backend ("auto", "marker", "docling", "markitdown")
    
    Example:
        pipeline = PaperPipeline(extraction_backend="marker", gpu=True)
        results = pipeline.search("transformer attention", sources=["arxiv"])
        paper = pipeline.process(results.papers[0])
        answer = pipeline.query("What is attention?")
    """
    
    def __init__(
        self,
        pdf_dir: str = None,
        markdown_dir: str = None,
        db_path: Optional[str] = None,
        vector_store: str = "chroma",
        embedding_model: str = "all-MiniLM-L6-v2",
        gpu: bool = False,
        extraction_backend: str = "auto",
        **kwargs
    ):
        if pdf_dir is None:
            pdf_dir = os.path.join(tempfile.gettempdir(), "paperflow", "pdfs")
        if markdown_dir is None:
            markdown_dir = os.path.join(tempfile.gettempdir(), "paperflow", "markdown")
        
        self.pdf_dir = Path(pdf_dir)
        self.markdown_dir = Path(markdown_dir)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        
        self._search = UnifiedSearch(**kwargs)
        self._extractor = PDFExtractor(backend=extraction_backend, gpu=gpu)
        self._chunker = TextChunker()
        
        self._embedder: Optional[EmbeddingProcessor] = None
        self._vector_store: Optional[VectorStoreAdapter] = None
        self._embedding_model = embedding_model
        self._vector_store_backend = vector_store
        self._db_path = db_path
        
        self._papers: Dict[str, Paper] = {}
    
    def search(
        self,
        query: str,
        sources: Optional[List[str | SourceType]] = None,
        max_results: int = 10,
        **kwargs
    ) -> SearchResult:
        """Search for papers across sources."""
        source_types = None
        if sources:
            source_types = [
                SourceType(s.lower()) if isinstance(s, str) else s
                for s in sources
            ]
        
        return self._search.search(
            query=query,
            sources=source_types,
            max_results=max_results,
            **kwargs
        )
    
    def download(self, paper: Paper | Dict[str, Any], pdf_dir: Optional[str] = None) -> Paper:
        """Download PDF for a paper.
        
        Args:
            paper: The paper to download (Paper object or paper dict)
            pdf_dir: Optional custom directory to save the PDF. If None, uses the pipeline's default pdf_dir.
        """
        if isinstance(paper, dict):
            # Convert dict to Paper object
            metadata = PaperMetadata(**paper)
            paper = Paper(metadata=metadata)
        elif isinstance(paper, PaperMetadata):
            paper = Paper(metadata=paper)
        
        paper.status = ProcessingStatus.DOWNLOADING
        
        # Use custom pdf_dir if provided, otherwise use self.pdf_dir
        save_dir = Path(pdf_dir) if pdf_dir else self.pdf_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        identifier = (
            paper.metadata.arxiv_id or
            paper.metadata.doi or
            paper.metadata.pmc_id or
            str(paper.uuid)[:8]
        )
        safe_id = identifier.replace("/", "_").replace(":", "_")
        pdf_path = save_dir / f"{safe_id}.pdf"
        
        provider = get_provider(paper.metadata.source)
        success = provider.download_pdf(paper.metadata.model_dump(), str(pdf_path))
        
        if success:
            paper.pdf_path = str(pdf_path)
            paper.has_pdf = True
            paper.status = ProcessingStatus.PENDING
            print(f"PDF saved to: {pdf_path}")
        else:
            paper.status = ProcessingStatus.FAILED
            paper.error_message = "PDF download failed"
        
        self._papers[str(paper.uuid)] = paper
        return paper
    
    def extract(self, paper: Paper) -> Paper:
        """Extract sections from paper PDF."""
        if not paper.pdf_path or not paper.has_pdf:
            paper.error_message = "No PDF available"
            return paper
        
        paper.status = ProcessingStatus.EXTRACTING
        
        if not self._extractor.available:
            if paper.metadata.abstract:
                paper.sections = [Section(
                    section_type=SectionType.ABSTRACT,
                    content=paper.metadata.abstract,
                    order=0
                )]
                paper.has_sections = True
            paper.status = ProcessingStatus.PENDING
            return paper
        
        sections_dict = self._extractor.extract_sections(paper.pdf_path)
        paper.sections = [
            Section(**section_data) for section_data in sections_dict.values()
        ]
        paper.has_sections = bool(paper.sections)
        paper.status = ProcessingStatus.PENDING
        
        if paper.sections:
            self._save_markdown(paper)
        
        self._papers[str(paper.uuid)] = paper
        return paper
    
    def chunk(self, paper: Paper) -> Paper:
        """Chunk paper sections for RAG."""
        paper.status = ProcessingStatus.CHUNKING
        
        if paper.sections:
            sections_dict = {s.section_type: s for s in paper.sections}
            paper.chunks = self._chunker.chunk_sections(sections_dict)
        elif paper.metadata.abstract:
            paper.chunks = self._chunker.chunk_text(
                paper.metadata.abstract,
                SectionType.ABSTRACT
            )
        
        paper.has_chunks = bool(paper.chunks)
        paper.status = ProcessingStatus.PENDING
        self._papers[str(paper.uuid)] = paper
        return paper
    
    def embed(self, paper: Paper) -> Paper:
        """Generate embeddings for paper chunks."""
        if not paper.chunks:
            return paper
        
        paper.status = ProcessingStatus.EMBEDDING
        
        if self._embedder is None:
            self._embedder = EmbeddingProcessor(model_name=self._embedding_model)
        
        if self._vector_store is None:
            kwargs = {"persist_directory": self._db_path} if self._db_path else {}
            self._vector_store = VectorStoreAdapter(
                backend=self._vector_store_backend, **kwargs
            )
        
        if not self._embedder.available:
            paper.error_message = "Embedding model not available"
            paper.status = ProcessingStatus.PENDING
            return paper
        
        results = self._embedder.embed_chunks(paper.chunks)
        embeddings = [emb for _, emb in results]
        
        self._vector_store.add(
            chunks=paper.chunks,
            embeddings=embeddings,
            paper_uuid=str(paper.uuid)
        )
        
        paper.has_embeddings = True
        paper.status = ProcessingStatus.COMPLETED
        self._papers[str(paper.uuid)] = paper
        return paper
    
    def process(
        self,
        paper: Paper | Dict[str, Any],
        download: bool = True,
        extract: bool = True,
        chunk: bool = True,
        embed: bool = False,
        pdf_dir: Optional[str] = None
    ) -> Paper:
        """Full processing pipeline.
        
        Args:
            paper: The paper to process
            download: Whether to download the PDF
            extract: Whether to extract text from PDF
            chunk: Whether to chunk the text
            embed: Whether to create embeddings
            pdf_dir: Optional custom directory to save the PDF. If None, uses the pipeline's default pdf_dir.
        """
        if isinstance(paper, PaperMetadata):
            paper = Paper(metadata=paper)
        elif isinstance(paper, dict):
            # Convert dict to Paper object
            metadata = PaperMetadata(**paper)
            paper = Paper(metadata=metadata)
        
        paper.citation = self._generate_citation(paper.metadata)
        
        if download:
            paper = self.download(paper, pdf_dir=pdf_dir)
            if paper.status == ProcessingStatus.FAILED:
                return paper
        
        if extract and paper.has_pdf:
            paper = self.extract(paper)
        
        if chunk:
            paper = self.chunk(paper)
        
        if embed and paper.has_chunks:
            paper = self.embed(paper)
        
        paper.updated_at = datetime.now()
        return paper
    
    def process_batch(
        self,
        papers: List[Paper | Dict[str, Any]],
        **kwargs
    ) -> List[Paper]:
        """Process multiple papers."""
        return [self.process(p, **kwargs) for p in papers]
    
    def query(
        self,
        question: str,
        paper_ids: Optional[List[str | UUID]] = None,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        RAG query across papers.
        
        Returns context chunks - integrate with LangChain/LlamaIndex for LLM.
        """
        if self._embedder is None or self._vector_store is None:
            return {"error": "Embeddings not initialized. Process papers with embed=True"}
        
        query_emb = self._embedder.embed_texts([question])[0]
        
        filter_paper = str(paper_ids[0]) if paper_ids and len(paper_ids) == 1 else None
        
        results = self._vector_store.search(
            query_embedding=query_emb,
            n_results=n_results,
            filter_paper=filter_paper
        )
        
        return {
            "question": question,
            "contexts": results,
            "paper_ids": [str(p) for p in paper_ids] if paper_ids else None
        }
    
    def get_paper(self, uuid: str | UUID) -> Optional[Paper]:
        """Get paper from cache."""
        return self._papers.get(str(uuid))
    
    def list_papers(self) -> List[Paper]:
        """List all processed papers."""
        return list(self._papers.values())
    
    def export_paper(self, paper: Paper, format: str = "json") -> str:
        """Export paper to JSON or dict."""
        if format == "json":
            return paper.model_dump_json(indent=2)
        return paper.to_dict()
    
    def _generate_citation(self, metadata: PaperMetadata) -> Citation:
        """Generate citations in multiple formats."""
        authors = metadata.get_author_names()
        year = metadata.year or "n.d."
        title = metadata.title
        
        # APA
        if len(authors) == 1:
            author_str = authors[0]
        elif len(authors) == 2:
            author_str = f"{authors[0]} & {authors[1]}"
        elif len(authors) > 2:
            author_str = f"{authors[0]} et al."
        else:
            author_str = "Unknown"
        
        if metadata.source == SourceType.ARXIV:
            apa = f"{author_str} ({year}). {title}. arXiv:{metadata.arxiv_id}."
        elif metadata.journal:
            apa = f"{author_str} ({year}). {title}. {metadata.journal}."
        else:
            apa = f"{author_str} ({year}). {title}."
        
        # BibTeX
        key = f"{authors[0].split()[-1].lower() if authors else 'unknown'}{year}"
        bibtex = f"""@article{{{key},
  title={{{title}}},
  author={{{' and '.join(authors)}}},
  year={{{year}}},
  {"journal={" + metadata.journal + "}," if metadata.journal else ""}
  {"doi={" + metadata.doi + "}," if metadata.doi else ""}
}}"""
        
        return Citation(apa=apa, bibtex=bibtex)
    
    def _save_markdown(self, paper: Paper) -> None:
        """Save paper as markdown."""
        safe_id = str(paper.uuid)[:8]
        md_path = self.markdown_dir / f"{safe_id}.md"
        
        content = f"# {paper.metadata.title}\n\n"
        content += f"**UUID:** `{paper.uuid}`\n\n"
        content += f"**Authors:** {', '.join(paper.metadata.get_author_names())}\n\n"
        content += f"**Year:** {paper.metadata.year}\n\n"
        content += f"**Source:** {paper.metadata.source.value}\n\n"
        
        if paper.citation:
            content += f"## Citation\n\n```\n{paper.citation.apa}\n```\n\n"
        
        for section in paper.sections:
            if section.section_type != SectionType.FULL_TEXT:
                content += f"## {section.section_type.value.title()}\n\n"
                content += f"{section.content}\n\n"
        
        md_path.write_text(content)
        paper.markdown_path = str(md_path)
