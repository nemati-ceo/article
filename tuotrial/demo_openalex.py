"""
PaperFlow Demo: OpenAlex Provider

Demonstrates searching and processing papers from OpenAlex,
an open catalog covering millions of scholarly works across all disciplines.

Usage:
    python demo_openalex.py
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
    print(f"   Open Access: {paper.get('is_open_access', 'N/A')}")
    
    if paper.get('abstract'):
        abstract = paper['abstract'][:150] + "..." if len(paper['abstract']) > 150 else paper['abstract']
        print(f"   Abstract: {abstract}")


def demo_basic_search(pipeline: PaperPipeline) -> dict:
    """Basic OpenAlex search demonstration."""
    print_header("Basic OpenAlex Search")
    
    query = "climate change adaptation strategies"
    print(f"Query: '{query}'")
    print("Searching...")
    
    results = pipeline.search(query, sources=["openalex"], max_results=5)
    
    print(f"[OK] Found {results.total_found} papers in {results.search_time_ms}ms")
    
    if results.papers:
        for i, paper in enumerate(results.papers, 1):
            print_paper_summary(paper, i)
    else:
        print("[WARNING] No papers found")
    
    return results


def demo_interdisciplinary_queries(pipeline: PaperPipeline) -> None:
    """Demonstrate queries across various disciplines."""
    print_header("Interdisciplinary Research Queries")
    
    queries = [
        ("Environmental Science", "renewable energy policy"),
        ("Economics", "behavioral economics decision making"),
        ("Social Science", "social media misinformation"),
        ("Engineering", "sustainable urban infrastructure")
    ]
    
    for discipline, query in queries:
        print(f"\n[{discipline}] Query: '{query}'")
        results = pipeline.search(query, sources=["openalex"], max_results=2)
        
        if results.papers:
            print(f"[OK] Found {results.total_found} papers")
            for paper in results.papers:
                title = paper['title'][:55] + "..." if len(paper['title']) > 55 else paper['title']
                year = paper.get('year', 'N/A')
                print(f"  - ({year}) {title}")
        else:
            print("[WARNING] No papers found")


def demo_open_access_papers(pipeline: PaperPipeline) -> None:
    """Find open access papers."""
    print_header("Open Access Papers")
    
    query = "machine learning healthcare"
    print(f"Query: '{query}'")
    print("Searching for papers (filtering for Open Access)...")
    
    results = pipeline.search(query, sources=["openalex"], max_results=10)
    
    if not results.papers:
        print("[WARNING] No papers found")
        return
    
    # Filter for open access
    oa_papers = [p for p in results.papers if p.get('is_open_access')]
    
    print(f"\nOpen Access papers: {len(oa_papers)}/{len(results.papers)}")
    
    for i, paper in enumerate(oa_papers[:5], 1):
        title = paper['title'][:50] + "..." if len(paper['title']) > 50 else paper['title']
        year = paper.get('year', 'N/A')
        citations = paper.get('citation_count', 0)
        print(f"  {i}. ({year}) {title} [{citations} citations]")


def demo_json_output(results) -> None:
    """Display results in JSON format."""
    print_header("JSON Output")
    
    if not results.papers:
        print("[WARNING] No papers to display")
        return
    
    print("First paper as JSON:")
    print(json.dumps(results.papers[0], indent=2))


def demo_download_and_extract(pipeline: PaperPipeline) -> None:
    """Download and extract content from an OpenAlex paper."""
    print_header("Download and Extract")
    
    print("Searching for open access papers...")
    results = pipeline.search("data science education", sources=["openalex"], max_results=5)
    
    if not results.papers:
        print("[ERROR] No papers found")
        return
    
    # Prefer open access papers
    oa_papers = [p for p in results.papers if p.get('is_open_access')]
    papers_to_try = oa_papers if oa_papers else results.papers
    
    # Try to find a paper with PDF available
    paper = None
    for paper_dict in papers_to_try[:3]:
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


def demo_process_batch(pipeline: PaperPipeline) -> None:
    """Process multiple OpenAlex papers."""
    print_header("Batch Processing")
    
    print("Searching for papers...")
    results = pipeline.search("artificial intelligence ethics", sources=["openalex"], max_results=3)
    
    if not results.papers:
        print("[ERROR] No papers found")
        return
    
    print(f"Processing {len(results.papers)} papers...\n")
    
    success_count = 0
    for i, paper_meta in enumerate(results.papers, 1):
        title = paper_meta['title'][:45] + "..." if len(paper_meta['title']) > 45 else paper_meta['title']
        oa_status = "OA" if paper_meta.get('is_open_access') else "Closed"
        print(f"[{i}/{len(results.papers)}] [{oa_status}] {title}")
        
        paper = pipeline.process(paper_meta, embed=False)
        
        if paper.has_pdf:
            success_count += 1
            print(f"    Status: OK")
            print(f"    Sections: {len(paper.sections)}")
            print(f"    Chunks: {len(paper.chunks)}")
        else:
            print(f"    Status: No PDF available")
        print()
    
    print(f"Summary: {success_count}/{len(results.papers)} papers processed with PDFs")


def main():
    """Main entry point."""
    print_header("PaperFlow: OpenAlex Provider Demo")
    
    # Configuration
    config = {
        "gpu": False,
        "pdf_dir": "./openalex_papers",
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
    demo_interdisciplinary_queries(pipeline)
    demo_open_access_papers(pipeline)
    demo_json_output(results)
    demo_download_and_extract(pipeline)
    demo_process_batch(pipeline)
    
    print_header("Demo Complete")
    print("OpenAlex demo finished successfully!")
    print("\nTip: OpenAlex has excellent Open Access coverage.")
    print("Filter for is_open_access=True for best PDF availability.")


if __name__ == "__main__":
    main()
