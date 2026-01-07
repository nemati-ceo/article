"""
Semantic Scholar provider implementation.
"""
import os
from typing import Any, Dict, List, Optional

import httpx

from paperflow.schemas import SourceType
from .base import BaseProvider


class SemanticScholarProvider(BaseProvider):
    """Provider for Semantic Scholar papers."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

    @property
    def source_type(self) -> SourceType:
        return SourceType.SEMANTIC_SCHOLAR

    @property
    def name(self) -> str:
        return "Semantic Scholar"

    def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Search Semantic Scholar for papers."""
        params = {
            "query": query,
            "limit": max_results,
            "fields": "paperId,title,authors,year,abstract,venue,publicationVenue,externalIds,url,openAccessPdf,isOpenAccess"
        }

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(f"{self.BASE_URL}/paper/search", params=params, headers=headers)
                response.raise_for_status()
                data = response.json()

            papers = []
            for paper_data in data.get("data", []):
                paper = self._convert_paper(paper_data)
                if self._passes_filters_dict(paper, **kwargs):
                    papers.append(paper)

            return papers

        except Exception as e:
            print(f"Semantic Scholar search error: {e}")
            return []

    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get paper by Semantic Scholar ID, DOI, or arXiv ID."""
        fields = "paperId,title,authors,year,abstract,venue,publicationVenue,externalIds,url,openAccessPdf,isOpenAccess,citationCount"

        params = {"fields": fields}
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(f"{self.BASE_URL}/paper/{paper_id}", params=params, headers=headers)
                response.raise_for_status()
                data = response.json()

            return self._convert_paper(data)

        except Exception as e:
            print(f"Semantic Scholar get_paper error: {e}")
            return None

    def download_pdf(self, paper: Dict[str, Any], output_path: str) -> bool:
        """Download PDF if open access URL available."""
        pdf_url = paper.get("pdf_url")
        if not pdf_url:
            return False

        try:
            with httpx.Client(timeout=120.0, follow_redirects=True) as client:
                response = client.get(pdf_url, headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status()

                with open(output_path, "wb") as f:
                    f.write(response.content)
                return True

        except Exception as e:
            print(f"Semantic Scholar download error: {e}")
            return False

    def _convert_paper(self, paper_data: dict) -> Dict[str, Any]:
        """Convert Semantic Scholar paper to dictionary."""
        authors = []
        for author_data in paper_data.get("authors", []):
            name = author_data.get("name", "")
            if name:
                authors.append({"name": name})

        external_ids = paper_data.get("externalIds", {})
        doi = external_ids.get("DOI")
        arxiv_id = external_ids.get("ArXiv")

        pdf_url = None
        oa = paper_data.get("openAccessPdf", {})
        if oa and oa.get("url"):
            pdf_url = oa["url"]

        venue = paper_data.get("venue", "") or paper_data.get("publicationVenue", {}).get("name", "")

        return {
            "title": paper_data.get("title", ""),
            "authors": authors,
            "year": paper_data.get("year"),
            "doi": doi,
            "arxiv_id": arxiv_id,
            "source": SourceType.SEMANTIC_SCHOLAR.value,
            "provider": self.name,
            "url": paper_data.get("url", ""),
            "pdf_url": pdf_url,
            "abstract": paper_data.get("abstract"),
            "citation_count": paper_data.get("citationCount"),
            "journal": venue,
            "published_date": None,  # Not provided in search
        }

    def _passes_filters_dict(self, paper: Dict[str, Any], **kwargs: Any) -> bool:
        """Check if paper passes filters."""
        if kwargs.get("year_from") and paper.get("year"):
            if paper["year"] < kwargs["year_from"]:
                return False
        if kwargs.get("year_to") and paper.get("year"):
            if paper["year"] > kwargs["year_to"]:
                return False
        return True