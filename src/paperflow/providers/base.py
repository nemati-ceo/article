"""
Base provider interface for all paper sources.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from paperflow.schemas import PaperMetadata, SearchQuery, SourceType


class BaseProvider(ABC):
    """Abstract base class for paper providers."""

    @property
    @abstractmethod
    def source_type(self) -> SourceType:
        """Return the source type for this provider."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any
    ) -> List[PaperMetadata]:
        """
        Search for papers.

        Args:
            query: Search query string
            max_results: Maximum number of results
            **kwargs: Provider-specific parameters

        Returns:
            List of PaperMetadata objects
        """
        pass

    @abstractmethod
    def get_paper(self, paper_id: str) -> Optional[PaperMetadata]:
        """
        Get a single paper by ID.

        Args:
            paper_id: Provider-specific paper ID (DOI, arXiv ID, PMID, etc.)

        Returns:
            PaperMetadata if found, None otherwise
        """
        pass

    @abstractmethod
    def download_pdf(self, paper: PaperMetadata, output_path: str) -> bool:
        """
        Download PDF for a paper.

        Args:
            paper: Paper metadata
            output_path: Path to save PDF

        Returns:
            True if successful, False otherwise
        """
        pass

    def search_from_query(self, query: SearchQuery) -> List[PaperMetadata]:
        """Search using SearchQuery object."""
        return self.search(
            query=query.query,
            max_results=query.max_results,
            year_from=query.year_from,
            year_to=query.year_to,
            author=query.author,
            categories=query.categories,
        )

    def _build_filters(self, **kwargs: Any) -> Dict[str, Any]:
        """Build provider-specific filters from kwargs."""
        filters = {}
        if kwargs.get("year_from"):
            filters["year_from"] = kwargs["year_from"]
        if kwargs.get("year_to"):
            filters["year_to"] = kwargs["year_to"]
        if kwargs.get("author"):
            filters["author"] = kwargs["author"]
        if kwargs.get("categories"):
            filters["categories"] = kwargs["categories"]
        return filters
