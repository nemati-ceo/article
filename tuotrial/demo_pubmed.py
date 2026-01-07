"""
PaperFlow Demo: PubMed Provider

Demonstrates searching and processing papers from PubMed,
the premier database for biomedical and life sciences literature.

Usage:
    python demo_pubmed.py
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
    print(f"   PMID: {paper.get('pmid', 'N/A')}")
    print(f"   Journal: {paper.get('journal', 'N/A')}")
    
    if paper.get('abstract'):
        abstract = paper['abstract'][:150] + "..." if len(paper['abstract']) > 150 else paper['abstract']
        print(f"   Abstract: {abstract}")


def demo_basic_search(pipeline: PaperPipeline) -> dict:
    """Basic PubMed search demonstration."""
    print_header("Basic PubMed Search")
    
    query = "CRISPR gene editing therapy"
    print(f"Query: '{query}'")
    print("Searching...")
    
    results = pipeline.search(query, sources=["pubmed"], max_results=5)
    
    print(f"[OK] Found {results.total_found} papers in {results.search_time_ms}ms")
    
    if results.papers:
        for i, paper in enumerate(results.papers, 1):
            print_paper_summary(paper, i)
    else:
        print("[WARNING] No papers found")
    
    return results


def demo_medical_queries(pipeline: PaperPipeline) -> None:
    """Demonstrate various medical/biomedical queries."""
    print_header("Medical Research Queries")
    
    queries = [
        "COVID-19 vaccine efficacy",
        "Alzheimer's disease biomarkers",
        "cancer immunotherapy checkpoint"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = pipeline.search(query, sources=["pubmed"], max_results=2)
        
        if results.papers:
            print(f"[OK] Found {results.total_found} papers")
            for paper in results.papers:
                title = paper['title'][:60] + "..." if len(paper['title']) > 60 else paper['title']
                print(f"  - {title}")
        else:
            print("[WARNING] No papers found")


def demo_json_output(results) -> None:
    """Display results in JSON format."""
    print_header("JSON Output")
    
    if not results.papers:
        print("[WARNING] No papers to display")
        return
    
    print("First paper as JSON:")
    print(json.dumps(results.papers[0], indent=2))


def demo_download_and_extract(pipeline: PaperPipeline) -> None:
    """Download and extract content from a PubMed paper."""
    print_header("Download and Extract")
    
    print("Searching for a paper with available PDF...")
    results = pipeline.search("machine learning clinical diagnosis", sources=["pubmed"], max_results=3)
    
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
        print("[WARNING] No PDF available (common for PubMed - many require subscriptions)")
        print("Tip: PubMed Central (PMC) papers often have free PDFs")
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
    """Process multiple PubMed papers."""
    print_header("Batch Processing")
    
    print("Searching for papers...")
    results = pipeline.search("diabetes mellitus treatment", sources=["pubmed"], max_results=3)
    
    if not results.papers:
        print("[ERROR] No papers found")
        return
    
    print(f"Processing {len(results.papers)} papers...\n")
    
    success_count = 0
    for i, paper_meta in enumerate(results.papers, 1):
        title = paper_meta['title'][:45] + "..." if len(paper_meta['title']) > 45 else paper_meta['title']
        print(f"[{i}/{len(results.papers)}] {title}")
        
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
    print_header("PaperFlow: PubMed Provider Demo")
    
    # Configuration
    config = {
        "gpu": False,
        "pdf_dir": "./pubmed_papers",
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
    demo_medical_queries(pipeline)
    demo_json_output(results)
    demo_download_and_extract(pipeline)
    demo_process_batch(pipeline)
    
    print_header("Demo Complete")
    print("PubMed demo finished successfully!")
    print("\nNote: PubMed PDF availability varies. For best results,")
    print("search for Open Access or PubMed Central (PMC) articles.")


if __name__ == "__main__":
    main()
