
# paperflow

Unified academic paper ingestion, extraction, and RAG pipeline.

## Features

- **Multi-Source Search**: Query arXiv, PubMed, Semantic Scholar, and OpenAlex from a single interface
- **PDF Download**: Automatic PDF retrieval from open-access sources
- **Structured Extraction**: Extract paper sections (abstract, introduction, methods, results, conclusion) using Marker AI
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

# With RAG support (embeddings + vector store)
pip install paperflow[rag]

# All features
pip install paperflow[all]
```

## Quick Start

```python
from paperflow import PaperPipeline

pipeline = PaperPipeline()

# Search across multiple sources
results = pipeline.search(
    "transformer attention mechanism",
    sources=["arxiv", "semantic_scholar"],
    max_results=10
)

# Process a paper (download → extract → chunk)
paper = pipeline.process(results.papers[0])

print(f"Title: {paper.metadata.title}")
print(f"Sections: {len(paper.sections)}")
print(f"Chunks: {len(paper.chunks)}")

# Export for RAG
docs = paper.to_langchain_documents()
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
```

### 1. Search

```python
from paperflow import PaperPipeline

pipeline = PaperPipeline()

# Single source
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
```

### 2. Download & Extract

```python
# Process single paper
paper = pipeline.process(results.papers[0])

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
from paperflow.src.providers import ArxivProvider, SemanticScholarProvider

# arXiv
arxiv = ArxivProvider()
papers = arxiv.search("BERT", max_results=10, categories=["cs.CL"])

# Semantic Scholar with recommendations
s2 = SemanticScholarProvider()
papers = s2.search("GPT-4", max_results=10)
recommendations = s2.get_recommendations(paper_id="some-paper-id")

# Download PDF
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
    embedding_model="all-MiniLM-L6-v2"  # Sentence transformer model
)
```

## Output Schemas

### Paper

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
└── src/
    ├── pipeline.py              # Main PaperPipeline class
    ├── schemas/
    │   └── paper.py             # Pydantic models
    ├── providers/
    │   ├── base.py              # Abstract base provider
    │   ├── arxiv_provider.py
    │   ├── pubmed_provider.py
    │   ├── semantic_scholar_provider.py
    │   └── openalex_provider.py
    └── processors/
        ├── marker_processor.py  # PDF extraction
        ├── chunker.py           # Text chunking
        └── embeddings.py        # Vector embeddings
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