"""
OpenAlex provider implementation.
Uses pyalex library if available, falls back to direct API.
"""
import os
from typing import Any, List, Optional

import httpx

from resarchflow.core.schemas import Author, PaperMetadata, SourceType
from .base import BaseProvider


class OpenAlexProvider(BaseProvider):
    """
    Provider for OpenAlex - free, open catalog of scholarly works.
    100,000 API calls/day without authentication.
    """
    
    BASE_URL = "https://api.openalex.org"
    
    def __init__(self, email: Optional[str] = None):
        self.email = email or os.getenv("OPENALEX_EMAIL", "")
        self._client: Optional[Any] = None
        self._use_library = self._try_import_library()
    
    def _try_import_library(self) -> bool:
        """Try to import pyalex library."""
        try:
            import pyalex
            if self.email:
                pyalex.config.email = self.email
            return True
        except ImportError:
            return False
    
    @property
    def source_type(self) -> SourceType:
        return SourceType.OPENALEX
    
    @property
    def name(self) -> str:
        return "OpenAlex"
    
    def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any
    ) -> List[PaperMetadata]:
        """Search OpenAlex for papers."""
        if self._use_library:
            return self._search_with_library(query, max_results, **kwargs)
        return self._search_with_api(query, max_results, **kwargs)
    
    def _search_with_library(
        self,
        query: str,
        max_results: int,
        **kwargs: Any
    ) -> List[PaperMetadata]:
        """Search using pyalex library."""
        try:
            from pyalex import Works
            
            works_query = Works().search(query)
            
            # Apply filters
            if kwargs.get("year_from"):
                works_query = works_query.filter(publication_year=f">{kwargs['year_from']-1}")
            if kwargs.get("year_to"):
                works_query = works_query.filter(publication_year=f"<{kwargs['year_to']+1}")
            
            results = works_query.get(per_page=max_results)
            
            papers = []
            for work in results:
                paper = self._convert_work(work)
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"OpenAlex library error: {e}")
            return []
    
    def _search_with_api(
        self,
        query: str,
        max_results: int,
        **kwargs: Any
    ) -> List[PaperMetadata]:
        """Search using direct API calls."""
        params = {
            "search": query,
            "per_page": max_results,
        }
        
        if self.email:
            params["mailto"] = self.email
        
        # Build filter string
        filters = []
        if kwargs.get("year_from"):
            filters.append(f"publication_year:>{kwargs['year_from']-1}")
        if kwargs.get("year_to"):
            filters.append(f"publication_year:<{kwargs['year_to']+1}")
        
        if filters:
            params["filter"] = ",".join(filters)
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    f"{self.BASE_URL}/works",
                    params=params
                )
                response.raise_for_status()
                data = response.json()
            
            papers = []
            for work in data.get("results", []):
                paper = self._convert_work(work)
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"OpenAlex API error: {e}")
            return []
    
    def get_paper(self, paper_id: str) -> Optional[PaperMetadata]:
        """
        Get paper by OpenAlex ID, DOI, or other identifier.
        
        paper_id can be:
        - OpenAlex ID: W2741809807
        - DOI: https://doi.org/10.1234/example
        - DOI short: 10.1234/example
        """
        # Format ID
        if paper_id.startswith("10."):
            paper_id = f"https://doi.org/{paper_id}"
        
        params = {}
        if self.email:
            params["mailto"] = self.email
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    f"{self.BASE_URL}/works/{paper_id}",
                    params=params
                )
                response.raise_for_status()
                data = response.json()
            
            return self._convert_work(data)
            
        except Exception as e:
            print(f"OpenAlex get_paper error: {e}")
            return None
    
    def download_pdf(self, paper: PaperMetadata, output_path: str) -> bool:
        """
        Download PDF if open access URL available.
        OpenAlex provides OA links from Unpaywall.
        """
        if not paper.pdf_url:
            return False
        
        try:
            with httpx.Client(timeout=120.0, follow_redirects=True) as client:
                response = client.get(
                    paper.pdf_url,
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                response.raise_for_status()
                
                with open(output_path, "wb") as f:
                    f.write(response.content)
                return True
                
        except Exception as e:
            print(f"OpenAlex download error: {e}")
            return False
    
    def _convert_work(self, work: dict) -> PaperMetadata:
        """Convert OpenAlex work to PaperMetadata."""
        # Extract authors
        authors = []
        for authorship in work.get("authorships", []):
            author_data = authorship.get("author", {})
            name = author_data.get("display_name", "")
            if name:
                affiliation = None
                institutions = authorship.get("institutions", [])
                if institutions:
                    affiliation = institutions[0].get("display_name")
                authors.append(Author(name=name, affiliation=affiliation))
        
        # Extract IDs
        ids = work.get("ids", {})
        doi = ids.get("doi", "").replace("https://doi.org/", "") if ids.get("doi") else None
        
        # Get PDF URL from open access info
        pdf_url = None
        oa = work.get("open_access", {})
        if oa.get("is_oa"):
            pdf_url = oa.get("oa_url")
        
        # Primary location for journal info
        primary_loc = work.get("primary_location", {}) or {}
        source = primary_loc.get("source", {}) or {}
        
        # Abstract (OpenAlex uses inverted index)
        abstract = None
        if work.get("abstract_inverted_index"):
            abstract = self._reconstruct_abstract(work["abstract_inverted_index"])
        
        return PaperMetadata(
            title=work.get("display_name", work.get("title", "")),
            authors=authors,
            year=work.get("publication_year"),
            doi=doi,
            source=SourceType.OPENALEX,
            url=work.get("id", ""),
            pdf_url=pdf_url,
            abstract=abstract,
            citation_count=work.get("cited_by_count"),
            journal=source.get("display_name"),
            published_date=work.get("publication_date"),
        )
    
    def _reconstruct_abstract(self, inverted_index: dict) -> str:
        """Reconstruct abstract from OpenAlex inverted index format."""
        if not inverted_index:
            return ""
        
        # Build list of (position, word) tuples
        words = []
        for word, positions in inverted_index.items():
            for pos in positions:
                words.append((pos, word))
        
        # Sort by position and join
        words.sort(key=lambda x: x[0])
        return " ".join(word for _, word in words)
