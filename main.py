"""
Main CLI for paper downloader.
"""
import sys
import time
from typing import Any, List, Optional

from paper_downloader.config import Config, get_config
from paper_downloader.database import PaperDatabase
from paper_downloader.extractor import MarkerExtractor
from paper_downloader.providers import ArxivProvider, PubMedProvider
from paper_downloader.providers.arxiv_provider import get_arxiv_filters_interactive
from paper_downloader.utils import display_arxiv_results, display_pubmed_results


def print_header(config: Config) -> None:
    """Print application header."""
    print("=" * 80)
    print("  📚 PAPER DOWNLOAD & EXTRACTION SYSTEM (Modular)")
    print("=" * 80)
    print(f"  📁 PDFs saved to: {config.pdf_folder}/")
    print(f"  📝 Markdown saved to: {config.markdown_folder}/")
    print(f"  🗄️  Metadata database: {config.metadata_file}")
    print("=" * 80)


def select_source() -> str:
    """Select paper source."""
    print("\n1. arXiv")
    print("2. PubMed (PMC Open Access)")
    choice = input("Select Source (1 or 2): ").strip()

    if choice not in ["1", "2"]:
        print("❌ Invalid choice. Please select 1 or 2.")
        sys.exit(1)

    return choice


def get_search_topic() -> str:
    """Get search topic from user."""
    topic = input("Enter search topic: ").strip()
    if not topic:
        print("❌ Search topic cannot be empty.")
        sys.exit(1)
    return topic


def search_papers(
    source: str, topic: str, config: Config
) -> tuple[Any, List[Any]]:
    """Search for papers and return provider and results."""
    if source == "1":
        provider = ArxivProvider()
        use_filters = input("Use advanced filters? (y/n): ").lower().strip()
        filters = get_arxiv_filters_interactive() if use_filters == "y" else None
        filters = filters or {}
        filters["max_results"] = config.max_results
    else:
        if not config.ncbi_email or not config.ncbi_api_key:
            print("⚠️ Warning: NCBI credentials not set in environment variables")
            print("   Set NCBI_EMAIL and NCBI_API_KEY for better rate limits")
        provider = PubMedProvider(
            config.ncbi_email, config.ncbi_api_key, config.max_results
        )
        filters = None

    print(f"\n🔍 Searching for '{topic}' (Max {config.max_results})...")
    papers = provider.search(topic, filters)

    if not papers:
        print("❌ No results found.")
        sys.exit(0)

    return provider, papers


def display_results(papers: List[Any], source: str) -> None:
    """Display search results based on source."""
    if source == "1":
        display_arxiv_results(papers)
    else:
        display_pubmed_results(papers)


def get_download_mode() -> str:
    """Get download mode from user."""
    print("\n📥 DOWNLOAD OPTIONS:")
    print("1. Download all PDFs only")
    print("2. Download all PDFs + Extract to Markdown")
    print("3. Select specific papers")
    print("4. Quit")
    return input("\nYour choice (1-4): ").strip()


def process_papers(
    papers: List[Any],
    provider: Any,
    db: PaperDatabase,
    config: Config,
    extractor: Optional[MarkerExtractor],
    extract_markdown: bool,
) -> None:
    """Process papers: download and optionally extract."""
    for paper in papers:
        success, paper_uuid = provider.download_and_save(
            paper, config.pdf_folder, db
        )

        if success and extract_markdown and paper_uuid and extractor:
            paper_data = db.get_paper(paper_uuid)
            if paper_data:
                print("   📄 Extracting sections from PDF...")
                sections = extractor.extract_sections(paper_data["pdf_path"])

                if sections:
                    md_path = extractor.create_markdown(
                        paper_uuid, paper_data, sections, config.markdown_folder
                    )
                    print(f"   ✅ Markdown saved: {md_path}")
                    db.update_paper(paper_uuid, {"markdown_path": md_path})

        time.sleep(config.download_delay)


def handle_selective_download(
    papers: List[Any],
    provider: Any,
    db: PaperDatabase,
    config: Config,
    extractor: Optional[MarkerExtractor],
) -> None:
    """Handle selective paper download."""
    indices_input = input("Enter paper numbers (comma-separated, e.g., 1,3,5): ").strip()
    try:
        selected = [int(x.strip()) - 1 for x in indices_input.split(",")]
        extract = input("Extract to markdown? (y/n): ").lower().strip() == "y"

        selected_papers = [papers[i] for i in selected if 0 <= i < len(papers)]
        process_papers(selected_papers, provider, db, config, extractor, extract)

    except ValueError:
        print("Invalid input")


def main() -> None:
    """Main entry point."""
    config = get_config()
    print_header(config)

    # Initialize extractor (loads Marker AI)
    extractor = MarkerExtractor()

    if not extractor.available:
        print("\n⚠️ WARNING: Marker AI not available!")
        print("   PDF downloads will work, but extraction will be skipped.")
        cont = input("\n   Continue anyway? (y/n): ").lower().strip()
        if cont != "y":
            sys.exit(0)

    db = PaperDatabase(config.metadata_file)
    source = select_source()
    topic = get_search_topic()

    provider, papers = search_papers(source, topic, config)
    display_results(papers, source)

    mode = get_download_mode()

    if mode == "1":
        print("\n🚀 Downloading PDFs only...")
        process_papers(papers, provider, db, config, None, False)
        print(f"\n✅ Complete! Check {config.pdf_folder}/ and {config.metadata_file}")

    elif mode == "2":
        print("\n🚀 Downloading PDFs and extracting to Markdown...")
        process_papers(papers, provider, db, config, extractor, True)
        print(f"\n✅ Complete! Check {config.pdf_folder}/, {config.markdown_folder}/")

    elif mode == "3":
        handle_selective_download(papers, provider, db, config, extractor)

    else:
        print("Exiting...")

    print("\n" + "=" * 80)
    print(f"📊 Total papers in database: {len(db)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
