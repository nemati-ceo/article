"""
PaperFlow Demo: Quickstart

A minimal example to get started with PaperFlow in under 2 minutes.

Usage:
    python demo_quickstart.py
"""

from paperflow import PaperPipeline


def main():
    # Create pipeline with minimal config
    pipeline = PaperPipeline(
        pdf_dir="./papers",
        extraction_backend="markitdown"
    )
    print("[OK] Pipeline initialized")
    
    # Search for papers
    print("\nSearching arXiv...")
    results = pipeline.search(
        "machine learning",
        sources=["arxiv"],
        max_results=3
    )
    print(f"[OK] Found {results.total_found} papers")
    
    # Display results
    print("\nResults:")
    for i, paper in enumerate(results.papers, 1):
        title = paper['title'][:60] + "..." if len(paper['title']) > 60 else paper['title']
        print(f"  {i}. {title}")
    
    # Process first paper
    if results.papers:
        print("\nProcessing first paper...")
        paper = pipeline.process(results.papers[0], embed=False)
        
        print(f"\n[OK] Done!")
        print(f"  PDF: {paper.pdf_path}")
        print(f"  Sections: {len(paper.sections)}")
        print(f"  Chunks: {len(paper.chunks)}")


if __name__ == "__main__":
    main()
