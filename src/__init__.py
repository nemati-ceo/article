"""
resarchflow - Unified paper ingestion, extraction, and RAG pipeline.

Example:
    from resarchflow import PaperPipeline
    
    pipeline = PaperPipeline()
    
    # Search across multiple sources
    results = pipeline.search(
        "transformer attention mechanism",
        sources=["arxiv", "semantic_scholar"],
        max_results=10
    )
    
    # Process papers (download, extract, chunk)
    for paper_meta in results.papers[:3]:
        paper = pipeline.process(paper_meta)
        print(f"Processed: {paper.metadata.title}")
        print(f"Sections: {len(paper.sections)}")
        print(f"Chunks: {len(paper.chunks)}")
    
    # Export for RAG
    for paper in pipeline.list_papers():
        docs = paper.to_langchain_documents()
"""
from resarchflow.core import (
    PaperPipeline,
    Paper,
    PaperMetadata,
    SearchResult,
    Section,
    Chunk,
    SourceType,
    SectionType,
    UnifiedSearch,
)

__version__ = "0.1.0"
__author__ = "Soodi"

__all__ = [
    "PaperPipeline",
    "Paper",
    "PaperMetadata",
    "SearchResult",
    "Section",
    "Chunk",
    "SourceType",
    "SectionType",
    "UnifiedSearch",
    "__version__",
]
