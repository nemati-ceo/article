"""
arXiv paper provider implementation.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import arxiv

from .base import BaseProvider


class ArxivProvider(BaseProvider):
    """Provider for arXiv papers."""

    @property
    def name(self) -> str:
        return "arXiv"

    def search(
        self, query: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """Search arXiv with optional filters."""
        filters = filters or {}
        search_query = self._build_query(query, filters)
        sort_criterion = self._get_sort_criterion(filters.get("sort_by", "announced"))
        max_results = filters.get("max_results", 50)

        client = arxiv.Client()
        search = arxiv.Search(
            query=search_query,
            max_results=max_results * 2,  # Fetch extra for filtering
            sort_by=sort_criterion,
            sort_order=arxiv.SortOrder.Descending,
        )

        papers = list(client.results(search))
        papers = self._apply_date_filter(papers, filters)

        return papers[:max_results]

    def _build_query(self, topic: str, filters: Dict[str, Any]) -> str:
        """Build advanced arXiv query with filters."""
        query_parts = [topic]

        if filters.get("categories"):
            cat_query = " OR ".join([f"cat:{cat}" for cat in filters["categories"]])
            query_parts.append(f"({cat_query})")

        if filters.get("author"):
            query_parts.append(f'au:{filters["author"]}')

        if filters.get("title_contains"):
            query_parts.append(f'ti:{filters["title_contains"]}')

        return " AND ".join(query_parts)

    def _get_sort_criterion(self, sort_by: str) -> arxiv.SortCriterion:
        """Get arxiv sort criterion from string."""
        criteria = {
            "announced": arxiv.SortCriterion.LastUpdatedDate,
            "submitted": arxiv.SortCriterion.SubmittedDate,
            "relevance": arxiv.SortCriterion.Relevance,
        }
        return criteria.get(sort_by, arxiv.SortCriterion.LastUpdatedDate)

    def _apply_date_filter(
        self, papers: List[Any], filters: Dict[str, Any]
    ) -> List[Any]:
        """Filter papers by date range."""
        if not (filters.get("date_from") or filters.get("date_to")):
            return papers

        filtered = []
        for paper in papers:
            paper_date = paper.published

            if filters.get("date_from") and paper_date < filters["date_from"]:
                continue
            if filters.get("date_to") and paper_date > filters["date_to"]:
                continue

            filtered.append(paper)

        return filtered

    def extract_metadata(self, paper: Any) -> Dict[str, Any]:
        """Extract all available metadata from arXiv paper."""
        return {
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "abstract": paper.summary,
            "year": paper.published.year,
            "published": paper.published.strftime("%Y-%m-%d"),
            "updated": paper.updated.strftime("%Y-%m-%d") if paper.updated else "N/A",
            "arxiv_id": paper.get_short_id(),
            "entry_id": paper.entry_id,
            "doi": paper.doi if paper.doi else "N/A",
            "primary_category": paper.primary_category,
            "categories": paper.categories,
            "comment": paper.comment if paper.comment else "N/A",
            "journal_ref": paper.journal_ref if paper.journal_ref else "N/A",
            "pdf_url": paper.pdf_url,
            "url": paper.entry_id,
            "source": "arXiv",
        }

    def download_pdf(self, paper: Any, pdf_path: str) -> bool:
        """Download PDF from arXiv."""
        try:
            paper.download_pdf(filename=pdf_path)
            return True
        except Exception as e:
            print(f"   ❌ arXiv download error: {e}")
            return False


def get_arxiv_filters_interactive() -> Dict[str, Any]:
    """Interactive filter selection for arXiv."""
    from paper_downloader.config import ARXIV_CATEGORIES

    filters: Dict[str, Any] = {}

    print("\n--- arXiv Filters (press Enter to skip) ---")

    # Sort selection
    print("\nSort by:")
    print("1. Latest Announcement (default)")
    print("2. Submission Date")
    print("3. Relevance")
    sort_choice = input("Choose (1-3): ").strip()

    sort_map = {"2": "submitted", "3": "relevance"}
    filters["sort_by"] = sort_map.get(sort_choice, "announced")

    # Category selection
    print("\nAvailable Categories:")
    cat_list = list(ARXIV_CATEGORIES.keys())
    for i, (code, name) in enumerate(ARXIV_CATEGORIES.items(), 1):
        print(f"{i}. {code} - {name}")

    cat_input = input("\nEnter category numbers (comma-separated, e.g., 1,2): ").strip()
    if cat_input:
        try:
            indices = [int(x.strip()) - 1 for x in cat_input.split(",")]
            filters["categories"] = [
                cat_list[i] for i in indices if 0 <= i < len(cat_list)
            ]
        except ValueError:
            print("Invalid input, skipping categories")

    # Date filter
    date_input = input("\nLast N days (e.g., 7 for last week): ").strip()
    if date_input:
        try:
            days = int(date_input)
            filters["date_from"] = datetime.now() - timedelta(days=days)
        except ValueError:
            print("Invalid input, skipping date filter")

    # Author filter
    author = input("\nFilter by author name: ").strip()
    if author:
        filters["author"] = author

    return filters
