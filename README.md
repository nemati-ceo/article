
# paperflow

Unified academic paper ingestion, extraction, and RAG pipeline with JSON-compatible API.

## Why JSON-Compatible?

Paperflow returns all search results as JSON-serializable dictionaries, making it perfect for:
- **Web APIs**: Direct serialization for REST endpoints
- **Data Pipelines**: Easy integration with ETL workflows
- **Frontend Apps**: Send results directly to web interfaces
- **Caching**: Store results in Redis, databases, or files
- **Cross-Language**: Use with JavaScript, Java, Go, etc.

Each result includes `provider` and `source` fields for easy attribution and filtering.

## Features

- **Multi-Source Search**: Query arXiv, PubMed, Semantic Scholar, and OpenAlex from a single interface
- **JSON-Compatible API**: All search results are JSON-serializable dictionaries with provider metadata
- **PDF Download**: Automatic PDF retrieval from open-access sources
- **Structured Extraction**: Extract paper sections (abstract, introduction, methods, results, conclusion) using Marker AI
- **GPU Acceleration**: Optional CUDA GPU support for faster PDF text extraction
- **RAG-Ready Output**: Pre-chunked text with metadata for direct use with LangChain, LlamaIndex, or custom pipelines
- **Vector Storage**: Built-in support for ChromaDB and in-memory vector stores
- **Citation Generation**: Auto-generate APA and BibTeX citations
- **LangChain Integration**: Export papers directly to LangChain Document format

## Installation

```bash
# Basic installation
pip install paperflow

# With PDF extraction (Marker AI)
pip install paperflow[extraction]


# All features
pip install paperflow[all]
```

## Quick Start

```python
from paperflow import PaperPipeline

# Create pipeline with GPU support (optional)
pipeline = PaperPipeline(
    gpu=True,                    # Enable GPU acceleration for PDF extraction
    extraction_backend="auto"    # PDF extraction backend: "auto", "marker", "docling", "markitdown"
)

# Search across multiple sources - returns JSON-compatible dictionaries
results = pipeline.search(
    "transformer attention mechanism",
    sources=["arxiv", "semantic_scholar"],
    max_results=10
)

# Each result is a JSON-serializable dictionary
paper_dict = results.papers[0]
print(f"Title: {paper_dict['title']}")
print(f"Provider: {paper_dict['provider']}")  # e.g., "arXiv"
print(f"Source: {paper_dict['source']}")      # e.g., "arxiv"

# Process a paper (download → extract → chunk)
paper = pipeline.process(paper_dict)  # Accepts both dicts and PaperMetadata

print(f"Sections: {len(paper.sections)}")
print(f"Chunks: {len(paper.chunks)}")

# Export for RAG
docs = paper.to_langchain_documents()
```

## Command Line Interface

Paperflow includes a command-line interface for quick searches:

```bash
# Install with CLI support
pip install paperflow

# Search and display results in a table
paperflow "transformer attention" --sources arxiv --max-results 5

# Search multiple sources
paperflow "machine learning" --sources arxiv pubmed openalex --max-results 10

# Enable GPU acceleration
paperflow "deep learning" --gpu --max-results 10
```

Example output:
```
Found 9 papers in 6641ms
Sources: ['arxiv', 'pubmed', 'openalex']

+-----+------------------------------------------+-------------------------------+--------+----------+-----------------+
|   # | Title                                    | Authors                       |   Year | Source   | Link/ID         |
+=====+==========================================+===============================+========+==========+=================+
|   1 | Changing Data Sources in the Age of      | Cedric De Boom, Michael       |   2023 | arxiv    | 2306.04338v1    |
|     | Machine Learning for Off...              | Reusens                       |        |          |                 |
+-----+------------------------------------------+-------------------------------+--------+----------+-----------------+
|   2 | Using Multiple Isotope-Labeled Infrared  | Bongalonta IJ, Dinner AR,     |   2025 | pubmed   | 10.1021/acs.jpc |
|     | Spectra for the Stru...                  | Tokmakoff A                   |        |          | b.5c05522       |
+-----+------------------------------------------+-------------------------------+--------+----------+-----------------+
```

## Supported Sources

| Source | Search | Download PDF | API Key Required |
|--------|--------|--------------|------------------|
| arXiv | ✅ | ✅ | No |
| PubMed/PMC | ✅ | ✅ (open access) | No (optional) |
| Semantic Scholar | ✅ | ❌ | No (optional) |
| OpenAlex | ✅ | ✅ (via Unpaywall) | No |

## Pipeline Stages

```
Search → Download → Extract → Chunk → Embed → Query
  🔍        ⬇️          🤖         ✂️        🧠        💾        💬
(JSON)     (PDF)      (Text)    (Chunks) (Vectors) (Store)   (RAG)
```

1. **Search**: Query multiple sources, get JSON results with provider metadata
2. **Download**: Fetch PDFs from open-access sources
3. **Extract**: Parse PDF text into structured sections using Marker AI
4. **Chunk**: Split text into RAG-optimized chunks
5. **Embed**: Generate vector embeddings for semantic search
6. **Query**: Answer questions using retrieved context

```python
from paperflow import PaperPipeline

pipeline = PaperPipeline()

# Single source - returns JSON-compatible dictionaries
results = pipeline.search("deep learning", sources=["arxiv"], max_results=20)

# Multiple sources with filters
results = pipeline.search(
    "machine learning healthcare",
    sources=["arxiv", "pubmed", "semantic_scholar", "openalex"],
    max_results=50,
    year_from=2020,
    year_to=2024
)

print(f"Found {results.total_found} papers from {len(results.sources_searched)} sources")

# Each paper is a JSON-serializable dictionary
for paper in results.papers[:3]:
    print(f"Title: {paper['title']}")
    print(f"Provider: {paper['provider']}")  # e.g., "arXiv", "PubMed", "OpenAlex"
    print(f"Source: {paper['source']}")      # e.g., "arxiv", "pubmed", "openalex"
    print("---")
```

### 2. Download & Extract

```python
# Process single paper - accepts both dictionaries and PaperMetadata objects
paper = pipeline.process(results.papers[0])  # results.papers[0] is a dict

# Access extracted sections
for section in paper.sections:
    print(f"{section.section_type.value}: {section.word_count} words")

# Access chunks
for chunk in paper.chunks:
    print(f"Chunk {chunk.index}: {len(chunk.content)} chars")
```

### 3. RAG Integration

```python
# With embeddings
paper = pipeline.process(results.papers[0], embed=True)

# Query across papers
context = pipeline.query("What is the attention mechanism?", n_results=5)
print(context["contexts"])

# Export to LangChain
docs = paper.to_langchain_documents()
# Returns: [{"page_content": "...", "metadata": {...}}, ...]
```

## Individual Providers

Use providers directly for more control:

```python
from paperflow.providers import ArxivProvider, PubMedProvider, OpenAlexProvider

# arXiv - returns JSON-compatible dictionaries
arxiv = ArxivProvider()
papers = arxiv.search("BERT", max_results=10, categories=["cs.CL"])

for paper in papers:
    print(f"Title: {paper['title']}")
    print(f"Provider: {paper['provider']}")  # "arXiv"
    print(f"Source: {paper['source']}")      # "arxiv"
    print(f"Year: {paper['year']}")
    print("---")

# PubMed
pubmed = PubMedProvider()
papers = pubmed.search("machine learning healthcare", max_results=5)

# OpenAlex
openalex = OpenAlexProvider()
papers = openalex.search("deep learning", max_results=5)

# Download PDF - accepts dictionary input
success = arxiv.download_pdf(papers[0], "paper.pdf")
```

## Text Processing

```python
from paperflow.src.processors import TextChunker, MarkerProcessor

# Extract sections from PDF
extractor = MarkerProcessor()
sections = extractor.extract_sections("paper.pdf")

# Chunk text for RAG
chunker = TextChunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk_sections(sections)
```

## Configuration

### Environment Variables

```bash
# Optional: PubMed API (increases rate limits)
export NCBI_EMAIL="your@email.com"
export NCBI_API_KEY="your_api_key"

# Optional: Semantic Scholar (increases rate limits)
export SEMANTIC_SCHOLAR_API_KEY="your_api_key"

# Optional: OpenAlex (polite pool access)
export OPENALEX_EMAIL="your@email.com"

# Optional: OpenAI embeddings
export OPENAI_API_KEY="your_api_key"
```

### Pipeline Options

```python
pipeline = PaperPipeline(
    pdf_dir="papers_pdf",           # PDF storage directory
    markdown_dir="papers_markdown", # Markdown output directory
    db_path="./chroma_db",          # Vector store persistence
    vector_store="chroma",          # "chroma" or "memory"
    embedding_model="all-MiniLM-L6-v2",  # Sentence transformer model
    gpu=True,                       # Enable GPU acceleration for PDF extraction
    extraction_backend="auto"       # PDF extraction backend: "auto", "marker", "docling", "markitdown"
)
```

## PDF Extraction Backends

PaperFlow supports multiple PDF extraction backends with different strengths:

| Backend | Quality | Speed | GPU Support | Table Extraction | Use Case |
|---------|---------|-------|-------------|------------------|----------|
| **Auto** | Variable | Variable | ✅ | Variable | **Recommended** - Automatic fallback |
| **Marker** | ⭐⭐⭐⭐⭐ | 🐌 | ✅ | ❌ | Best for academic papers, high accuracy |
| **Docling** | ⭐⭐⭐⭐ | 🐌 | ✅ | ✅ | Good table/figure extraction, IBM |
| **MarkItDown** | ⭐⭐⭐ | ⚡ | ❌ | ❌ | Lightweight, fast, CPU only |

### Backend Selection

```python
# Auto-selection (recommended) - tries Marker → Docling → MarkItDown
pipeline = PaperPipeline(extraction_backend="auto", gpu=True)

# High quality academic papers
pipeline = PaperPipeline(extraction_backend="marker", gpu=True)

# Tables and figures extraction
pipeline = PaperPipeline(extraction_backend="docling", gpu=True)

# Fast processing, CPU only
pipeline = PaperPipeline(extraction_backend="markitdown")
```

## Output Schemas

### Search Results (JSON-Compatible Dictionaries)

All search operations return JSON-serializable dictionaries with consistent structure:

```python
{
    "title": "Attention Is All You Need",
    "authors": [{"name": "Ashish Vaswani"}, {"name": "Noam Shazeer"}],
    "year": 2017,
    "doi": "10.48550/arXiv.1706.03762",
    "arxiv_id": "1706.03762",
    "source": "arxiv",
    "provider": "arXiv",
    "url": "https://arxiv.org/abs/1706.03762",
    "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
    "abstract": "The dominant sequence transduction models...",
    "citation_count": 50000,
    "journal": null,
    "categories": ["cs.CL", "cs.LG"]
}
```

### Paper Object (After Processing)

```python
Paper(
    uuid="...",
    metadata=PaperMetadata(...),
    sections=[Section(...)],
    chunks=[Chunk(...)],
    citation=Citation(apa="...", bibtex="..."),
    status="completed",
    has_pdf=True,
    has_sections=True,
    has_chunks=True,
    has_embeddings=False
)
```

### PaperMetadata

```python
PaperMetadata(
    title="Attention Is All You Need",
    authors=[Author(name="Ashish Vaswani", affiliation="Google")],
    year=2017,
    doi="10.48550/arXiv.1706.03762",
    arxiv_id="1706.03762",
    source="arxiv",
    url="https://arxiv.org/abs/1706.03762",
    abstract="The dominant sequence transduction models...",
    citation_count=50000
)
```

## Project Structure

```
paperflow/
├── __init__.py
├── cli.py                         # Command-line interface
├── pipeline.py                    # Main PaperPipeline class
├── schemas/
│   ├── __init__.py
│   └── paper.py                   # Pydantic models
├── providers/
│   ├── __init__.py
│   ├── base.py                    # Abstract base provider
│   ├── arxiv_provider.py          # arXiv search & download
│   ├── pubmed_provider.py         # PubMed/PMC search & download
│   ├── semantic_scholar_provider.py # Semantic Scholar (arXiv API)
│   └── openalex_provider.py       # OpenAlex search & download
└── processors/
    ├── __init__.py
    ├── marker_processor.py        # PDF text extraction
    ├── chunker.py                 # Text chunking
    └── embeddings.py              # Vector embeddings
```
```

## Requirements

- Python >= 3.9
- pydantic >= 2.0
- httpx >= 0.25.0
- arxiv >= 2.0.0
- biopython >= 1.80

### Optional Dependencies

- **extraction**: marker-pdf
- **rag**: langchain, chromadb, sentence-transformers
- **providers**: pyalex, semanticscholar

## PDF Extraction Backends

PaperFlow supports multiple PDF extraction backends with different strengths:

| Backend | Quality | Speed | GPU Support | Table Extraction | Use Case |
|---------|---------|-------|-------------|------------------|----------|
| **Marker** | ⭐⭐⭐⭐⭐ | 🐌 | ✅ | ❌ | Best for academic papers, high accuracy |
| **Docling** | ⭐⭐⭐⭐ | 🐌 | ✅ | ✅ | Good table/figure extraction, IBM |
| **MarkItDown** | ⭐⭐⭐ | ⚡ | ❌ | ❌ | Lightweight, fast, Microsoft |
| **Auto** | Variable | Variable | ✅ | Variable | Automatic fallback: Marker → Docling → MarkItDown |

### Installation Options

```bash
# Lightweight extraction (fastest, lowest quality)
pip install paperflow[extraction-light]

# Full extraction with Docling (tables, figures)
pip install paperflow[extraction-docling]

# All backends (best quality, largest install)
pip install paperflow[extraction-all]
```

### Usage Examples

#### Easy Pipeline Usage (Recommended)

```python
from paperflow import PaperPipeline

# Create pipeline with your preferred backend
pipeline = PaperPipeline(extraction_backend="auto", gpu=True)

# Process papers automatically
results = pipeline.search("machine learning", sources=["arxiv"])
paper = pipeline.process(results.papers[0])  # Downloads, extracts, chunks, embeds
```

#### Advanced Direct Usage

```python
from paperflow.processors.marker_processor import PDFExtractor

# Auto-select best available backend
extractor = PDFExtractor(backend="auto", gpu=True)

# Force specific backend
extractor = PDFExtractor(backend="marker", gpu=True)      # High quality
extractor = PDFExtractor(backend="docling", gpu=True)     # Tables/figures  
extractor = PDFExtractor(backend="markitdown")            # Fast, CPU only

# Extract content
text = extractor.extract_full_text("paper.pdf")
sections = extractor.extract_sections("paper.pdf")
content = extractor.extract_with_tables("paper.pdf")  # Docling only
```

### Backend Selection Guide

- **Academic Papers**: Use `marker` for highest quality text extraction
- **Tables/Charts**: Use `docling` for structured content extraction
- **Quick Processing**: Use `markitdown` for speed
- **Production**: Use `auto` for automatic fallback and reliability

## License

MIT


# Summary - paperflow Library

```
paperflow/
├── pyproject.toml                    # (keep your existing one, update name)
├── __init__.py                       # ← paperflow__init__.py
└── src/
    ├── __init__.py                   # ← src__init__.py
    ├── pipeline.py                   # ← pipeline.py
    ├── schemas/
    │   ├── __init__.py               # ← schemas/__init__.py
    │   └── paper.py                  # ← schemas/paper.py
    ├── providers/
    │   ├── __init__.py               # ← providers/__init__.py
    │   ├── base.py                   # ← providers/base.py  ✅ HERE
    │   ├── arxiv_provider.py
    │   ├── pubmed_provider.py
    │   ├── semantic_scholar_provider.py
    │   └── openalex_provider.py
    └── processors/
        ├── __init__.py
        ├── marker_processor.py
        ├── chunker.py
        └── embeddings.py     
```

```md
┌─────────────────────────────────────────────────────────────────────────────┐
│                           paperflow ARCHITECTURE                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER (Django REST)                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ /search/    │ │ /download/  │ │ /extract/   │ │ /query/     │           │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────────────────┐
│                              SERVICE LAYER                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │PaperService │ │SearchService│ │ExtractSvc   │ │ RAGService  │           │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
│  PROVIDER LAYER   │   │   PROCESSOR LAYER     │   │    WORKER LAYER       │
│                   │   │                       │   │                       │
│ ┌───────────────┐ │   │ ┌───────────────────┐ │   │ ┌───────────────────┐ │
│ │ ArxivProvider │ │   │ │ MarkerProcessor   │ │   │ │ Celery Worker     │ │
│ ├───────────────┤ │   │ ├───────────────────┤ │   │ ├───────────────────┤ │
│ │ PubMedProvider│ │   │ │ SectionExtractor  │ │   │ │ DownloadTask      │ │
│ ├───────────────┤ │   │ ├───────────────────┤ │   │ ├───────────────────┤ │
│ │ SemanticSchol.│ │   │ │ ChunkProcessor    │ │   │ │ ExtractTask       │ │
│ ├───────────────┤ │   │ ├───────────────────┤ │   │ ├───────────────────┤ │
│ │ OpenAlexProv. │ │   │ │ EmbeddingProcessor│ │   │ │ EmbedTask         │ │
│ ├───────────────┤ │   │ └───────────────────┘ │   │ └───────────────────┘ │
│ │ PaperScraper  │ │   │                       │   │                       │
│ └───────────────┘ │   └───────────────────────┘   └───────────────────────┘
└───────────────────┘                                           
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STORAGE LAYER                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ PostgreSQL  │ │ChromaDB/    │ │   Redis     │ │  S3/MinIO   │           │
│  │ (metadata)  │ │FAISS(vector)│ │  (cache)    │ │  (files)    │           │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘


DATA FLOW:
══════════
  Search ──▶ Download ──▶ Extract ──▶ Chunk ──▶ Embed ──▶ Store ──▶ Query
    🔍          ⬇️          🤖         ✂️        🧠        💾        💬


PROJECT STRUCTURE:
══════════════════
paperflow/
├── core/                    # Standalone pip package
│   ├── providers/           # arxiv, pubmed, semantic_scholar, openalex
│   ├── processors/          # marker, sections, chunker, embeddings
│   ├── storage/             # database, vector_store
│   ├── schemas/             # Pydantic models (RAG-ready output)
│   └── pipeline.py          # Main orchestrator
├── django_app/              # Optional Django integration
│   ├── papers/              # models, views, serializers, tasks
│   └── api/                 # REST endpoints
└── notebooks/               # Jupyter tutorials