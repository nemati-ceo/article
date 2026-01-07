#!/usr/bin/env python3
"""
Test script for paperflow package.
Run from the project root directory.

Usage:
    cd paperflow/   (where pyproject.toml is)
    python tests/test_local.py
"""
import sys
from pathlib import Path

# Add the project root to path so imports work without pip install
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """Test that all imports work."""
    print("=" * 60)
    print("TEST 1: Imports")
    print("=" * 60)
    
    try:
        from paperflow.schemas import (
            Paper,
            PaperMetadata,
            SearchResult,
            SourceType,
            SectionType,
        )
        print("✅ Schema imports OK")
    except ImportError as e:
        print(f"❌ Schema import error: {e}")
        return False
    
    try:
        from paperflow.providers import (
            ArxivProvider,
            PubMedProvider,
            SemanticScholarProvider,
            OpenAlexProvider,
            UnifiedSearch,
        )
        print("✅ Provider imports OK")
    except ImportError as e:
        print(f"❌ Provider import error: {e}")
        return False
    
    try:
        from paperflow.processors import (
            MarkerProcessor,
            TextChunker,
            EmbeddingProcessor,
        )
        print("✅ Processor imports OK")
    except ImportError as e:
        print(f"❌ Processor import error: {e}")
        return False
    
    try:
        from paperflow.pipeline import PaperPipeline
        print("✅ Pipeline import OK")
    except ImportError as e:
        print(f"❌ Pipeline import error: {e}")
        return False
    
    return True


def test_version():
    """Test that version is accessible."""
    print("\n" + "=" * 60)
    print("TEST 1.5: Version")
    print("=" * 60)
    
    try:
        import paperflow
        version = paperflow.__version__
        print(f"✅ Version: {version}")
        return True
    except Exception as e:
        print(f"❌ Version error: {e}")
        return False


def test_schema_creation():
    """Test that schemas can be created."""
    print("\n" + "=" * 60)
    print("TEST 2: Schema Creation")
    print("=" * 60)
    
    from paperflow.schemas import (
        Author,
        PaperMetadata,
        Paper,
        Section,
        Chunk,
        SourceType,
        SectionType,
    )
    
    # Create Author
    author = Author(name="John Doe", affiliation="MIT")
    print(f"✅ Author: {author.name}")
    
    # Create PaperMetadata
    metadata = PaperMetadata(
        title="Test Paper",
        authors=[author, "Jane Smith"],
        year=2024,
        doi="10.1234/test",
        source=SourceType.ARXIV,
        url="https://arxiv.org/abs/2401.00001",
        abstract="This is a test abstract.",
    )
    print(f"✅ PaperMetadata: {metadata.title}")
    print(f"   Authors: {metadata.get_author_names()}")
    
    # Create Paper
    paper = Paper(metadata=metadata)
    print(f"✅ Paper UUID: {paper.uuid}")
    
    # Create Section
    section = Section(
        section_type=SectionType.ABSTRACT,
        content="This is the abstract content.",
    )
    print(f"✅ Section: {section.section_type.value} ({section.word_count} words)")
    
    # Create Chunk
    chunk = Chunk(
        section_type=SectionType.ABSTRACT,
        content="This is a chunk.",
        index=0,
    )
    print(f"✅ Chunk ID: {chunk.chunk_id}")
    
    return True


def test_providers_init():
    """Test that providers can be initialized."""
    print("\n" + "=" * 60)
    print("TEST 3: Provider Initialization")
    print("=" * 60)
    
    from paperflow.providers import (
        ArxivProvider,
        PubMedProvider,
        SemanticScholarProvider,
        OpenAlexProvider,
    )
    
    try:
        arxiv = ArxivProvider()
        print(f"✅ {arxiv.name} provider initialized")
    except Exception as e:
        print(f"❌ ArxivProvider error: {e}")
    
    try:
        pubmed = PubMedProvider()
        print(f"✅ {pubmed.name} provider initialized")
    except Exception as e:
        print(f"❌ PubMedProvider error: {e}")
    
    try:
        s2 = SemanticScholarProvider()
        print(f"✅ {s2.name} provider initialized")
    except Exception as e:
        print(f"❌ SemanticScholarProvider error: {e}")
    
    try:
        oa = OpenAlexProvider()
        print(f"✅ {oa.name} provider initialized")
    except Exception as e:
        print(f"❌ OpenAlexProvider error: {e}")
    
    return True


def test_pipeline_init():
    """Test that pipeline can be initialized."""
    print("\n" + "=" * 60)
    print("TEST 4: Pipeline Initialization")
    print("=" * 60)
    
    from paperflow.pipeline import PaperPipeline
    import tempfile
    import os
    
    try:
        # Test with custom directories
        pipeline_custom = PaperPipeline(
            pdf_dir="test_pdfs",
            markdown_dir="test_markdown",
        )
        print("✅ Pipeline initialized with custom dirs")
        print(f"   PDF dir: {pipeline_custom.pdf_dir}")
        print(f"   Markdown dir: {pipeline_custom.markdown_dir}")
        
        # Test with default (temp) directories
        pipeline_temp = PaperPipeline()
        print("✅ Pipeline initialized with temp dirs")
        print(f"   PDF dir: {pipeline_temp.pdf_dir}")
        print(f"   Markdown dir: {pipeline_temp.markdown_dir}")
        
        # Verify temp dirs are in system temp
        temp_base = tempfile.gettempdir()
        assert str(pipeline_temp.pdf_dir).startswith(temp_base), f"PDF dir not in temp: {pipeline_temp.pdf_dir}"
        assert str(pipeline_temp.markdown_dir).startswith(temp_base), f"Markdown dir not in temp: {pipeline_temp.markdown_dir}"
        assert "paperflow" in str(pipeline_temp.pdf_dir), f"PDF dir missing paperflow: {pipeline_temp.pdf_dir}"
        assert "paperflow" in str(pipeline_temp.markdown_dir), f"Markdown dir missing paperflow: {pipeline_temp.markdown_dir}"
        
        print("✅ Temp directory paths validated")
        return True
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chunker():
    """Test text chunking."""
    print("\n" + "=" * 60)
    print("TEST 5: Text Chunker")
    print("=" * 60)
    
    from paperflow.processors import TextChunker
    from paperflow.schemas import SectionType
    
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    text = """
    This is the first paragraph of text. It contains several sentences 
    that will be chunked together. The chunker should handle this well.
    
    This is the second paragraph. It also has multiple sentences. 
    We want to see how the chunker splits this content into pieces.
    
    And here is a third paragraph for good measure. More content means
    more chunks to test with. Let's see what happens.
    """
    
    chunks = chunker.chunk_text(text, SectionType.ABSTRACT)
    
    print(f"✅ Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i}: {len(chunk.content)} chars")
    
    return True


def test_langchain_export():
    """Test LangChain document export."""
    print("\n" + "=" * 60)
    print("TEST 6: LangChain Export")
    print("=" * 60)
    
    from paperflow.schemas import (
        Paper, PaperMetadata, Chunk, SourceType, SectionType
    )
    
    metadata = PaperMetadata(
        title="Test Paper for LangChain",
        authors=["Author One"],
        year=2024,
        source=SourceType.ARXIV,
        url="https://example.com",
    )
    
    paper = Paper(metadata=metadata)
    paper.chunks = [
        Chunk(section_type=SectionType.ABSTRACT, content="Abstract text", index=0),
        Chunk(section_type=SectionType.INTRODUCTION, content="Intro text", index=1),
    ]
    
    docs = paper.to_langchain_documents()
    
    print(f"✅ Exported {len(docs)} LangChain documents")
    for doc in docs:
        print(f"   Section: {doc['metadata']['section']}")
        print(f"   Content: {doc['page_content'][:30]}...")
    
    return True


def test_arxiv_search():
    """Test actual arXiv search (requires internet)."""
    print("\n" + "=" * 60)
    print("TEST 7: arXiv Search (requires internet)")
    print("=" * 60)
    
    from paperflow.pipeline import PaperPipeline
    
    try:
        pipeline = PaperPipeline()
        results = pipeline.search(
            "attention transformer",
            sources=["arxiv"],
            max_results=3,
        )
        
        print(f"✅ Search completed in {results.search_time_ms}ms")
        print(f"   Found: {results.total_found} papers")
        
        for i, paper in enumerate(results.papers[:3], 1):
            title = paper["title"][:60] + "..." if len(paper["title"]) > 60 else paper["title"]
            print(f"\n   {i}. {title}")
            print(f"      Year: {paper.get('year', 'N/A')}")
            print(f"      arXiv: {paper.get('arxiv_id', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Search error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_semantic_scholar_search():
    """Test Semantic Scholar search (requires internet)."""
    print("\n" + "=" * 60)
    print("TEST 8: Semantic Scholar Search (requires internet)")
    print("=" * 60)
    
    from paperflow.providers import SemanticScholarProvider
    
    try:
        provider = SemanticScholarProvider()
        papers = provider.search("BERT language model", max_results=3)
        
        print(f"✅ Found {len(papers)} papers")
        for i, paper in enumerate(papers[:3], 1):
            title = paper.title[:50] + "..." if len(paper.title) > 50 else paper.title
            print(f"   {i}. {title} ({paper.year})")
        
        return True
    except Exception as e:
        print(f"❌ Semantic Scholar error: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("paperflow TEST SUITE")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 60)
    
    results = []
    
    # Core tests (no internet required)
    results.append(("Imports", test_imports()))
    
    if not results[-1][1]:
        print("\n❌ Import failed. Fix imports before continuing.")
        print("   Make sure you're running from project root:")
        print("   cd paperflow/")
        print("   python tests/test_local.py")
        return 1
    
    results.append(("Version", test_version()))
    results.append(("Schema Creation", test_schema_creation()))
    results.append(("Provider Init", test_providers_init()))
    results.append(("Pipeline Init", test_pipeline_init()))
    results.append(("Text Chunker", test_chunker()))
    results.append(("LangChain Export", test_langchain_export()))
    
    # Network tests
    print("\n" + "=" * 60)
    print("NETWORK TESTS (require internet)")
    print("=" * 60)
    
    try:
        results.append(("arXiv Search", test_arxiv_search()))
        results.append(("Semantic Scholar", test_semantic_scholar_search()))
    except Exception as e:
        print(f"⚠️ Network tests skipped: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to push to GitHub.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please fix before pushing.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())