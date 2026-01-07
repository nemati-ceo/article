"""
Pydantic schemas for paperflow.
These define the RAG-ready output format.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Supported paper sources."""
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    OPENALEX = "openalex"
    CROSSREF = "crossref"
    BIORXIV = "biorxiv"
    MEDRXIV = "medrxiv"


class ProcessingStatus(str, Enum):
    """Paper processing status."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    COMPLETED = "completed"
    FAILED = "failed"


class SectionType(str, Enum):
    """Standard paper sections."""
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    BACKGROUND = "background"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    FULL_TEXT = "full_text"


class Author(BaseModel):
    """Author information."""
    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None
    orcid: Optional[str] = None


class PaperMetadata(BaseModel):
    """Paper metadata from any source."""
    title: str
    authors: List[Author | str]
    year: Optional[int] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmid: Optional[str] = None
    pmc_id: Optional[str] = None
    source: SourceType
    url: str
    pdf_url: Optional[str] = None
    abstract: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    published_date: Optional[datetime] = None
    citation_count: Optional[int] = None

    def get_author_names(self) -> List[str]:
        """Get list of author names."""
        names = []
        for author in self.authors:
            if isinstance(author, str):
                names.append(author)
            else:
                names.append(author.name)
        return names


class Section(BaseModel):
    """Extracted paper section."""
    section_type: SectionType
    title: Optional[str] = None
    content: str
    order: int = 0
    word_count: int = 0

    def model_post_init(self, __context: Any) -> None:
        """Calculate word count after init."""
        if self.word_count == 0:
            self.word_count = len(self.content.split())


class Chunk(BaseModel):
    """Text chunk for RAG."""
    chunk_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    section_type: SectionType
    content: str
    index: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Generated citation."""
    apa: str
    mla: Optional[str] = None
    bibtex: Optional[str] = None
    chicago: Optional[str] = None


class Paper(BaseModel):
    """Complete paper object - RAG ready."""
    uuid: UUID = Field(default_factory=uuid4)
    metadata: PaperMetadata
    sections: List[Section] = Field(default_factory=list)
    chunks: List[Chunk] = Field(default_factory=list)
    citation: Optional[Citation] = None

    pdf_path: Optional[str] = None
    markdown_path: Optional[str] = None

    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    has_pdf: bool = False
    has_sections: bool = False
    has_chunks: bool = False
    has_embeddings: bool = False

    def to_langchain_documents(self) -> List[Dict[str, Any]]:
        """Convert to LangChain Document format."""
        docs = []
        for chunk in self.chunks:
            docs.append({
                "page_content": chunk.content,
                "metadata": {
                    "paper_uuid": str(self.uuid),
                    "title": self.metadata.title,
                    "authors": self.metadata.get_author_names(),
                    "year": self.metadata.year,
                    "source": self.metadata.source.value,
                    "section": chunk.section_type.value,
                    "chunk_index": chunk.index,
                    **chunk.metadata
                }
            })
        return docs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return self.model_dump(mode="json")


class SearchQuery(BaseModel):
    """Search query parameters."""
    query: str
    sources: List[SourceType] = Field(default_factory=lambda: [SourceType.ARXIV])
    max_results: int = 10
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    author: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    sort_by: str = "relevance"


class SearchResult(BaseModel):
    """Search result container."""
    query: SearchQuery
    papers: List[Dict[str, Any]]
    total_found: int
    sources_searched: List[SourceType]
    search_time_ms: int = 0

    def __str__(self) -> str:
        """Return a tabular representation of the search results."""
        from tabulate import tabulate

        if not self.papers:
            return f"Found {self.total_found} papers in {self.search_time_ms}ms\nSources: {[s.value for s in self.sources_searched]}\n\nNo papers found."

        # Prepare table data
        table_data = []
        for i, paper in enumerate(self.papers, 1):
            authors = []
            for author in paper.get("authors", []):
                if isinstance(author, dict):
                    authors.append(author.get("name", ""))
                else:
                    authors.append(str(author))
            authors_str = ", ".join(authors[:3])  # Limit to 3 authors
            if len(authors) > 3:
                authors_str += " et al."
            title = paper.get("title", "")[:60] + "..." if len(paper.get("title", "")) > 60 else paper.get("title", "")
            year = paper.get("year") or "N/A"
            source = paper.get("source", "unknown")
            # Add link/ID column
            link_id = paper.get("arxiv_id") or paper.get("doi") or paper.get("url", "")
            if link_id and len(link_id) > 50:
                link_id = link_id[:47] + "..."
            table_data.append([i, title, authors_str, year, source, link_id or "N/A"])

        table = tabulate(
            table_data,
            headers=["#", "Title", "Authors", "Year", "Source", "Link/ID"],
            tablefmt="grid",
            maxcolwidths=[3, 40, 30, 4, 8, 15]
        )

        summary = f"Found {self.total_found} papers in {self.search_time_ms}ms\nSources: {[s.value for s in self.sources_searched]}"
        return f"{summary}\n\n{table}"
