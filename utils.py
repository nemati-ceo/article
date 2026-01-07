"""
Utility functions for paper downloader.
"""
import re
from typing import Any, List


def sanitize_filename(title: str, max_length: int = 50) -> str:
    """Create safe filename from title."""
    safe = re.sub(r"[^\w\s-]", "", title)
    safe = re.sub(r"[-\s]+", "_", safe)
    return safe[:max_length]


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text with suffix if exceeds max_length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def display_arxiv_results(papers: List[Any]) -> None:
    """Display arXiv search results in table format."""
    print("\n" + "=" * 140)
    print(
        f"{'#':<4} | {'arXiv ID':<15} | {'Published':<12} | "
        f"{'Updated':<12} | {'Categories':<20} | {'Title':<50}"
    )
    print("=" * 140)

    for i, paper in enumerate(papers):
        arxiv_id = paper.get_short_id()
        published = paper.published.strftime("%Y-%m-%d")
        updated = paper.updated.strftime("%Y-%m-%d") if paper.updated else "N/A"
        
        categories = ", ".join(paper.categories[:2])
        categories = truncate_text(categories, 18)
        
        title = paper.title.replace("\n", " ")
        title = truncate_text(title, 47)

        print(
            f"{i+1:<4} | {arxiv_id:<15} | {published:<12} | "
            f"{updated:<12} | {categories:<20} | {title:<50}"
        )

    print("=" * 140 + "\n")


def display_pubmed_results(papers: List[dict]) -> None:
    """Display PubMed search results in table format."""
    print("\n" + "=" * 130)
    print(
        f"{'#':<4} | {'PMC ID':<15} | {'Date':<12} | "
        f"{'Source':<25} | {'Title':<50}"
    )
    print("=" * 130)

    for i, paper in enumerate(papers):
        pmc_id = paper["pmc_id"]
        date = str(paper["date"])[:12] if paper["date"] else "N/A"
        source = truncate_text(paper["source"], 23) if paper["source"] else "N/A"
        title = paper["title"].replace("\n", " ")
        title = truncate_text(title, 47)

        print(
            f"{i+1:<4} | {pmc_id:<15} | {date:<12} | "
            f"{source:<25} | {title:<50}"
        )

    print("=" * 130 + "\n")
