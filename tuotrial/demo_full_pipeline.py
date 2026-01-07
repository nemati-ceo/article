"""
PaperFlow Demo: Full Pipeline

Demonstrates the complete paper processing workflow:
Search -> Download -> Extract -> Chunk -> Embed -> Query

This demo shows how to build a RAG (Retrieval-Augmented Generation)
system with academic papers.

Usage:
    python demo_full_pipeline.py
"""

import json
from paperflow import PaperPipeline


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_step(step_num: int, title: str) -> None:
    """Print a step header."""
    print(f"\n[Step {step_num}] {title}")
    print("-" * 40)


def demo_step_by_step(pipeline: PaperPipeline) -> None:
    """Demonstrate each pipeline step individually."""
    print_header("Step-by-Step Pipeline")
    print("Pipeline: Search -> Download -> Extract -> Chunk -> Embed")
    
    # Step 1: Search
    print_step(1, "Search")
    query = "transformer neural network architecture"
    print(f"Query: '{query}'")
    
    results = pipeline.search(query, sources=["arxiv"], max_results=1)
    
    if not results.papers:
        print("[ERROR] No papers found")
        return None
    
    paper_dict = results.papers[0]
    print(f"[OK] Found: {paper_dict['title'][:60]}...")
    print(f"     Authors: {', '.join([a['name'] for a in paper_dict['authors'][:2]])}")
    print(f"     Year: {paper_dict.get('year', 'N/A')}")
    
    # Step 2: Download
    print_step(2, "Download PDF")
    paper = pipeline.download(paper_dict)
    
    if not paper.has_pdf:
        print(f"[ERROR] Download failed: {paper.error_message}")
        return None
    
    print(f"[OK] PDF saved to: {paper.pdf_path}")
    print(f"     File exists: {paper.has_pdf}")
    
    # Step 3: Extract
    print_step(3, "Extract Content")
    paper = pipeline.extract(paper)
    
    print(f"[OK] Extracted {len(paper.sections)} sections")
    
    if paper.sections:
        print("\nSections found:")
        for section in paper.sections[:5]:
            content_len = len(section.content)
            print(f"  - {section.section_type.value}: {content_len} chars")
    
    # Step 4: Chunk
    print_step(4, "Create Chunks")
    paper = pipeline.chunk(paper)
    
    print(f"[OK] Created {len(paper.chunks)} chunks")
    
    if paper.chunks:
        print("\nSample chunks:")
        for i, chunk in enumerate(paper.chunks[:3], 1):
            preview = chunk.content[:60] + "..." if len(chunk.content) > 60 else chunk.content
            print(f"  {i}. [{chunk.section_type.value}] {preview}")
    
    # Step 5: Embed
    print_step(5, "Generate Embeddings")
    try:
        paper = pipeline.embed(paper)
        print(f"[OK] Embeddings created: {paper.has_embeddings}")
        print(f"     Status: {paper.status.value}")
    except Exception as e:
        print(f"[SKIP] Embedding failed: {type(e).__name__}")
        print("       Install sentence-transformers for embeddings")
    
    return paper


def demo_single_call_process(pipeline: PaperPipeline) -> None:
    """Demonstrate processing with single method call."""
    print_header("Single-Call Processing")
    print("Use pipeline.process() to run all steps at once")
    
    # Search
    print("\nSearching for paper...")
    results = pipeline.search("BERT language model", sources=["arxiv"], max_results=1)
    
    if not results.papers:
        print("[ERROR] No papers found")
        return
    
    paper_dict = results.papers[0]
    print(f"Selected: {paper_dict['title'][:50]}...")
    
    # Process all at once
    print("\nProcessing (download + extract + chunk)...")
    paper = pipeline.process(paper_dict, embed=False)
    
    print(f"\n[OK] Processing complete!")
    print(f"     Has PDF: {paper.has_pdf}")
    print(f"     Sections: {len(paper.sections)}")
    print(f"     Chunks: {len(paper.chunks)}")
    print(f"     Status: {paper.status.value}")


def demo_batch_processing(pipeline: PaperPipeline) -> None:
    """Demonstrate batch processing of multiple papers."""
    print_header("Batch Processing")
    
    print("Searching for multiple papers...")
    results = pipeline.search("deep learning optimization", sources=["arxiv"], max_results=3)
    
    if not results.papers:
        print("[ERROR] No papers found")
        return
    
    print(f"Found {len(results.papers)} papers\n")
    
    # Process batch
    papers = pipeline.process_batch(results.papers, embed=False)
    
    # Summary
    print("\nBatch Results:")
    print("-" * 50)
    
    success = sum(1 for p in papers if p.has_pdf)
    total_sections = sum(len(p.sections) for p in papers)
    total_chunks = sum(len(p.chunks) for p in papers)
    
    for i, paper in enumerate(papers, 1):
        title = paper.metadata.title[:40] + "..." if len(paper.metadata.title) > 40 else paper.metadata.title
        status = "OK" if paper.has_pdf else "FAILED"
        print(f"  {i}. [{status}] {title}")
    
    print(f"\nSummary:")
    print(f"  Papers with PDF: {success}/{len(papers)}")
    print(f"  Total sections: {total_sections}")
    print(f"  Total chunks: {total_chunks}")


def demo_rag_query(pipeline: PaperPipeline) -> None:
    """Demonstrate RAG query functionality."""
    print_header("RAG Query Demo")
    
    print("Building knowledge base...")
    
    # Search and process papers
    results = pipeline.search("attention mechanism transformer", sources=["arxiv"], max_results=2)
    
    if not results.papers:
        print("[ERROR] No papers found")
        return
    
    # Process with embeddings
    processed_papers = []
    for paper_dict in results.papers:
        try:
            paper = pipeline.process(paper_dict, embed=True)
            if paper.has_embeddings:
                processed_papers.append(paper)
                print(f"[OK] Indexed: {paper.metadata.title[:40]}...")
        except Exception as e:
            print(f"[SKIP] Could not embed: {type(e).__name__}")
    
    if not processed_papers:
        print("[WARNING] No papers with embeddings. Install sentence-transformers.")
        return
    
    # Query
    print("\nQuerying knowledge base...")
    question = "What is the attention mechanism?"
    print(f"Question: '{question}'")
    
    response = pipeline.query(question, n_results=3)
    
    if "error" in response:
        print(f"[ERROR] {response['error']}")
        return
    
    print(f"\n[OK] Found {len(response['contexts'])} relevant chunks:")
    for i, ctx in enumerate(response['contexts'][:3], 1):
        preview = ctx['content'][:100] + "..." if len(ctx['content']) > 100 else ctx['content']
        print(f"\n  {i}. {preview}")


def demo_export_paper(pipeline: PaperPipeline) -> None:
    """Demonstrate paper export functionality."""
    print_header("Export Paper")
    
    # Get a processed paper
    results = pipeline.search("convolutional neural network", sources=["arxiv"], max_results=1)
    
    if not results.papers:
        print("[ERROR] No papers found")
        return
    
    paper = pipeline.process(results.papers[0], embed=False)
    
    # Export to JSON
    print("Exporting paper to JSON...")
    json_output = pipeline.export_paper(paper, format="json")
    
    # Show partial output
    data = json.loads(json_output)
    print(f"\nExported fields:")
    for key in data.keys():
        print(f"  - {key}")
    
    print(f"\nCitation (APA):")
    if data.get('citation'):
        print(f"  {data['citation'].get('apa', 'N/A')}")


def main():
    """Main entry point."""
    print_header("PaperFlow: Full Pipeline Demo")
    
    # Configuration
    config = {
        "gpu": False,
        "pdf_dir": "./pipeline_papers",
        "extraction_backend": "markitdown",
        "embedding_model": "all-MiniLM-L6-v2"
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = PaperPipeline(**config)
    print("[OK] Pipeline ready")
    
    # Run demos
    demo_step_by_step(pipeline)
    demo_single_call_process(pipeline)
    demo_batch_processing(pipeline)
    demo_rag_query(pipeline)
    demo_export_paper(pipeline)
    
    print_header("Demo Complete")
    print("Full pipeline demo finished!")
    print("\nNext steps:")
    print("  1. Install sentence-transformers for embeddings")
    print("  2. Use pipeline.query() for RAG applications")
    print("  3. Export to LangChain for advanced workflows")


if __name__ == "__main__":
    main()
