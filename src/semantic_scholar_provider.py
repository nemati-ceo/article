"""
Semantic Scholar provider implementation.
Uses the official semanticscholar library if available,
falls back to direct API calls.
"""
import os
from typing import Any, List, Optional

import httpx

from resarchflow.core.schemas import Author, PaperMetadata, SourceType
from .base import BaseProvider


class SemanticScholarProvider(BaseProvider):
    """Provider for Semantic Scholar papers."""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        self._client: Optional[Any] = None
        self._use_library = self._try_import_library()
    
    def _try_import_library(self) -> bool:
        """Try to import semanticscholar library."""
        try:
            from semanticscholar import SemanticScholar
            self._client = SemanticScholar(api_key=self.api_key) if self.api_key else SemanticScholar()
            return True
        except ImportError:
            return False
    
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
    ) -> List[PaperMetadata]:
        """Search Semantic Scholar for papers."""
        if self._use_library:
            return self._search_with_library(query, max_results, **kwargs)
        return self._search_with_api(query, max_results, **kwargs)
    
    def _search_with_library(
        self,
        query: str,
        max_results: int,
        **kwargs: Any
    ) -> List[PaperMetadata]:
        """Search using semanticscholar library."""
        try:
            results = self._client.search_paper(
                query,
                limit=max_results,
                fields=[
                    "title", "authors", "year", "abstract",
                    "externalIds", "url", "citationCount",
                    "publicationDate", "journal"
                ]
            )
            
            papers = []
            for result in results:
                paper = self._convert_library_result(result)
                if self._passes_filters(paper, **kwargs):
                    papers.append(paper)
            
            return papers[:max_results]
            
        except Exception as e:
            print(f"Semantic Scholar library error: {e}")
            return []
    
    def _search_with_api(
        self,
        query: str,
        max_results: int,
        **kwargs: Any
    ) -> List[PaperMetadata]:
        """Search using direct API calls."""
        fields = "title,authors,year,abstract,externalIds,url,citationCount,publicationDate,journal"
        
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    f"{self.BASE_URL}/paper/search",
                    params={
                        "query": query,
                        "limit": max_results,
                        "fields": fields
                    },
                    headers=headers
                )
                response.raise_for_status()
                data = response.json()
            
            papers = []
            for item in data.get("data", []):
                paper = self._convert_api_result(item)
                if self._passes_filters(paper, **kwargs):
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Semantic Scholar API error: {e}")
            return []
    
    def get_paper(self, paper_id: str) -> Optional[PaperMetadata]:
        """Get paper by Semantic Scholar ID, DOI, or arXiv ID."""
        fields = "title,authors,year,abstract,externalIds,url,citationCount,publicationDate,journal"
        
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    f"{self.BASE_URL}/paper/{paper_id}",
                    params={"fields": fields},
                    headers=headers
                )
                response.raise_for_status()
                data = response.json()
            
            return self._convert_api_result(data)
            
        except Exception as e:
            print(f"Semantic Scholar get_paper error: {e}")
            return None
    
    def download_pdf(self, paper: PaperMetadata, output_path: str) -> bool:
        """
        Semantic Scholar doesn't host PDFs directly.
        Returns False - use other providers for PDF download.
        """
        return False
    
    def get_recommendations(
        self,
        paper_id: str,
        max_results: int = 10
    ) -> List[PaperMetadata]:
        """Get paper recommendations based on a paper."""
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}",
                    params={"limit": max_results, "fields": "title,authors,year,abstract,externalIds,url"},
                    headers=headers
                )
                response.raise_for_status()
                data = response.json()
            
            return [
                self._convert_api_result(item)
                for item in data.get("recommendedPapers", [])
            ]
            
        except Exception as e:
            print(f"Semantic Scholar recommendations error: {e}")
            return []
    
    def _convert_library_result(self, result: Any) -> PaperMetadata:
        """Convert library result to PaperMetadata."""
        external_ids = result.externalIds or {}
        
        authors = []
        if result.authors:
            for a in result.authors:
                authors.append(Author(name=a.name if hasattr(a, 'name') else str(a)))
        
        return PaperMetadata(
            title=result.title or "",
            authors=authors,
            year=result.year,
            doi=external_ids.get("DOI"),
            arxiv_id=external_ids.get("ArXiv"),
            pmid=external_ids.get("PubMed"),
            source=SourceType.SEMANTIC_SCHOLAR,
            url=result.url or "",
            abstract=result.abstract,
            citation_count=result.citationCount,
            journal=result.journal.get("name") if result.journal else None,
        )
    
    def _convert_api_result(self, data: dict) -> PaperMetadata:
        """Convert API result to PaperMetadata."""
        external_ids = data.get("externalIds") or {}
        
        authors = []
        for a in data.get("authors", []):
            authors.append(Author(name=a.get("name", "")))
        
        journal_data = data.get("journal") or {}
        
        return PaperMetadata(
            title=data.get("title", ""),
            authors=authors,
            year=data.get("year"),
            doi=external_ids.get("DOI"),
            arxiv_id=external_ids.get("ArXiv"),
            pmid=external_ids.get("PubMed"),
            source=SourceType.SEMANTIC_SCHOLAR,
            url=data.get("url", ""),
            abstract=data.get("abstract"),
            citation_count=data.get("citationCount"),
            journal=journal_data.get("name") if isinstance(journal_data, dict) else None,
        )
    
    def _passes_filters(self, paper: PaperMetadata, **kwargs: Any) -> bool:
        """Check if paper passes filters."""
        if kwargs.get("year_from") and paper.year:
            if paper.year < kwargs["year_from"]:
                return False
        if kwargs.get("year_to") and paper.year:
            if paper.year > kwargs["year_to"]:
                return False
        return True
