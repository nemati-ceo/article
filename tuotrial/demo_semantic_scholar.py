"""
PaperFlow Demo: Semantic Scholar Provider

Demonstrates searching and processing papers from Semantic Scholar,
an AI-powered academic search engine with citation analysis.

Usage:
    python demo_semantic_scholar.py
"""

import json
from paperflow import PaperPipeline


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_paper_summary(paper: dict, index: int) -> None:
    """Print a formatted paper summary."""
    title = paper['title'][:65] + "..." if len(paper['title']) > 65 else paper['title']
    authors = ", ".join([a['name'] for a in paper['authors'][:2]])
    if len(paper['authors']) > 2:
        authors += " et al."
    
    print(f"\n{index}. {title}")
    print(f"   Authors: {authors}")
    print(f"   Year: {paper.get('year', 'N/A')}")
    print(f"   DOI: {paper.get('doi', 'N/A')}")
    print(f"   Citations: {paper.get('citation_count', 0)}")
    print(f"   Venue: {paper.get('venue', 'N/A')}")
    
    if paper.get('abstract'):
        abstract = paper['abstract'][:150] + "..." if len(paper['abstract']) > 150 else paper['abstract']
        print(f"   Abstract: {abstract}")


def demo_basic_search(pipeline: PaperPipeline) -> dict:
    """Basic Semantic Scholar search demonstration."""
    print_header("Basic Semantic Scholar Search")
    
    query = "large language models GPT"
    print(f"Query: '{query}'")
    print("Searching...")
    
    results = pipeline.search(query, sources=["semantic_scholar"], max_results=5)
    
    print(f"[OK] Found {results.total_found} papers in {results.search_time_ms}ms")
    
    if results.papers:
        for i, paper in enumerate(results.papers, 1):
            print_paper_summary(paper, i)
    else:
        print("[WARNING] No papers found")
    
    return results


def demo_ai_ml_queries(pipeline: PaperPipeline) -> None:
    """Demonstrate AI/ML research queries."""
    print_header("AI/ML Research Queries")
    
    queries = [
        "transformer architecture attention",
        "reinforcement learning from human feedback",
        "diffusion models image generation"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = pipeline.search(query, sources=["semantic_scholar"], max_results=2)
        
        if results.papers:
            print(f"[OK] Found {results.total_found} papers")
            for paper in results.papers:
                title = paper['title'][:55] + "..." if len(paper['title']) > 55 else paper['title']
                citations = paper.get('citation_count', 0)
                print(f"  - {title} [{citations} citations]")
        else:
            print("[WARNING] No papers found")


def demo_high_citation_papers(pipeline: PaperPipeline) -> None:
    """Find highly cited papers on a topic."""
    print_header("High Citation Papers")
    
    query = "BERT language model"
    print(f"Query: '{query}'")
    print("Searching and sorting by citations...")
    
    results = pipeline.search(query, sources=["semantic_scholar"], max_results=5)
    
    if not results.papers:
        print("[WARNING] No papers found")
        return
    
    # Sort by citation count
    sorted_papers = sorted(
        results.papers,
        key=lambda p: p.get('citation_count', 0),
        reverse=True
    )
    
    print("\nTop cited papers:")
    for i, paper in enumerate(sorted_papers[:5], 1):
        title = paper['title'][:50] + "..." if len(paper['title']) > 50 else paper['title']
        citations = paper.get('citation_count', 0)
        year = paper.get('year', 'N/A')
        print(f"  {i}. [{citations:,} citations] ({year}) {title}")


def demo_json_output(results) -> None:
    """Display results in JSON format."""
    print_header("JSON Output")
    
    if not results.papers:
        print("[WARNING] No papers to display")
        return
    
    print("First paper as JSON:")
    print(json.dumps(results.papers[0], indent=2))


def demo_download_and_extract(pipeline: PaperPipeline) -> None:
    """Download and extract content from a Semantic Scholar paper."""
    print_header("Download and Extract")
    
    print("Searching for a paper...")
    results = pipeline.search("neural network optimization", sources=["semantic_scholar"], max_results=3)
    
    if not results.papers:
        print("[ERROR] No papers found")
        return
    
    # Try to find a paper with PDF available
    paper = None
    for paper_dict in results.papers:
        print(f"Trying: {paper_dict['title'][:40]}...")
        paper = pipeline.download(paper_dict)
        if paper.has_pdf:
            break
    
    if not paper or not paper.has_pdf:
        print("[WARNING] No PDF available for these papers")
        return
    
    print(f"[OK] PDF saved to: {paper.pdf_path}")
    
    # Extract
    print("\nExtracting content...")
    paper = pipeline.extract(paper)
    print(f"[OK] Extracted {len(paper.sections)} sections")
    
    # Chunk
    print("\nCreating chunks...")
    paper = pipeline.chunk(paper)
    print(f"[OK] Created {len(paper.chunks)} chunks")
    
    # Show sample chunks
    if paper.chunks:
        print("\nSample chunks:")
        for chunk in paper.chunks[:3]:
            preview = chunk.content[:80] + "..." if len(chunk.content) > 80 else chunk.content
            print(f"  - [{chunk.section_type.value}] {preview}")


def demo_process_batch(pipeline: PaperPipeline) -> None:
    """Process multiple Semantic Scholar papers."""
    print_header("Batch Processing")
    
    print("Searching for papers...")
    results = pipeline.search("computer vision object detection", sources=["semantic_scholar"], max_results=3)
    
    if not results.papers:
        print("[ERROR] No papers found")
        return
    
    print(f"Processing {len(results.papers)} papers...\n")
    
    for i, paper_meta in enumerate(results.papers, 1):
        title = paper_meta['title'][:45] + "..." if len(paper_meta['title']) > 45 else paper_meta['title']
        print(f"[{i}/{len(results.papers)}] {title}")
        
        paper = pipeline.process(paper_meta, embed=False)
        
        status = "OK" if paper.has_pdf else "No PDF"
        print(f"    Status: {status}")
        print(f"    Sections: {len(paper.sections)}")
        print(f"    Chunks: {len(paper.chunks)}")
        print()


def main():
    """Main entry point."""
    print_header("PaperFlow: Semantic Scholar Provider Demo")
    
    # Configuration
    config = {
        "gpu": False,
        "pdf_dir": "./semantic_scholar_papers",
        "extraction_backend": "markitdown"
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = PaperPipeline(**config)
    print("[OK] Pipeline ready")
    
    # Run demos
    results = demo_basic_search(pipeline)
    demo_ai_ml_queries(pipeline)
    demo_high_citation_papers(pipeline)
    demo_json_output(results)
    demo_download_and_extract(pipeline)
    demo_process_batch(pipeline)
    
    print_header("Demo Complete")
    print("Semantic Scholar demo finished successfully!")


if __name__ == "__main__":
    main()
