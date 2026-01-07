"""
PaperFlow Demo: arXiv Provider

Demonstrates searching and processing papers from arXiv,
the preprint server for physics, mathematics, computer science, and more.

Usage:
    python demo_arxiv.py
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
    print(f"   arXiv ID: {paper.get('arxiv_id', 'N/A')}")
    print(f"   Citations: {paper.get('citation_count', 0)}")
    
    if paper.get('abstract'):
        abstract = paper['abstract'][:150] + "..." if len(paper['abstract']) > 150 else paper['abstract']
        print(f"   Abstract: {abstract}")


def demo_basic_search(pipeline: PaperPipeline) -> dict:
    """Basic arXiv search demonstration."""
    print_header("Basic arXiv Search")
    
    query = "transformer attention mechanism"
    print(f"Query: '{query}'")
    print("Searching...")
    
    results = pipeline.search(query, sources=["arxiv"], max_results=5)
    
    print(f"[OK] Found {results.total_found} papers in {results.search_time_ms}ms")
    
    if results.papers:
        for i, paper in enumerate(results.papers, 1):
            print_paper_summary(paper, i)
    else:
        print("[WARNING] No papers found")
    
    return results


def demo_json_output(results) -> None:
    """Display results in JSON format."""
    print_header("JSON Output")
    
    if not results.papers:
        print("[WARNING] No papers to display")
        return
    
    print("First paper as JSON:")
    print(json.dumps(results.papers[0], indent=2))


def demo_download_and_extract(pipeline: PaperPipeline) -> None:
    """Download and extract content from an arXiv paper."""
    print_header("Download and Extract")
    
    print("Searching for a paper to process...")
    results = pipeline.search("deep learning neural networks", sources=["arxiv"], max_results=1)
    
    if not results.papers:
        print("[ERROR] No papers found")
        return
    
    paper_dict = results.papers[0]
    print(f"Selected: {paper_dict['title'][:50]}...")
    
    # Download
    print("\n[Step 1] Downloading PDF...")
    paper = pipeline.download(paper_dict)
    
    if not paper.has_pdf:
        print(f"[ERROR] Download failed: {paper.error_message}")
        return
    
    print(f"[OK] PDF saved to: {paper.pdf_path}")
    
    # Extract
    print("\n[Step 2] Extracting content...")
    paper = pipeline.extract(paper)
    print(f"[OK] Extracted {len(paper.sections)} sections")
    
    # Display sections
    if paper.sections:
        print("\nExtracted sections:")
        for section in paper.sections[:5]:
            content_preview = section.content[:80] + "..." if len(section.content) > 80 else section.content
            print(f"  - {section.section_type.value}: {content_preview}")
    
    # Chunk
    print("\n[Step 3] Creating chunks...")
    paper = pipeline.chunk(paper)
    print(f"[OK] Created {len(paper.chunks)} chunks")


def demo_process_multiple(pipeline: PaperPipeline) -> None:
    """Process multiple arXiv papers."""
    print_header("Batch Processing")
    
    print("Searching for papers...")
    results = pipeline.search("reinforcement learning", sources=["arxiv"], max_results=3)
    
    if not results.papers:
        print("[ERROR] No papers found")
        return
    
    print(f"Processing {len(results.papers)} papers...\n")
    
    for i, paper_meta in enumerate(results.papers, 1):
        title = paper_meta['title'][:45] + "..." if len(paper_meta['title']) > 45 else paper_meta['title']
        print(f"[{i}/{len(results.papers)}] {title}")
        
        paper = pipeline.process(paper_meta, embed=False)
        
        status = "OK" if paper.has_pdf else "FAILED"
        print(f"    Status: {status}")
        print(f"    Sections: {len(paper.sections)}")
        print(f"    Chunks: {len(paper.chunks)}")
        print()


def main():
    """Main entry point."""
    print_header("PaperFlow: arXiv Provider Demo")
    
    # Configuration
    config = {
        "gpu": False,
        "pdf_dir": "./arxiv_papers",
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
    demo_json_output(results)
    demo_download_and_extract(pipeline)
    demo_process_multiple(pipeline)
    
    print_header("Demo Complete")
    print("arXiv demo finished successfully!")


if __name__ == "__main__":
    main()
