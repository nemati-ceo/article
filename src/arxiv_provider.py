"""
arXiv provider implementation.
"""
from datetime import datetime
from typing import Any, List, Optional

import arxiv

from resarchflow.core.schemas import Author, PaperMetadata, SourceType
from .base import BaseProvider


class ArxivProvider(BaseProvider):
    """Provider for arXiv papers."""
    
    def __init__(self):
        self.client = arxiv.Client()
    
    @property
    def source_type(self) -> SourceType:
        return SourceType.ARXIV
    
    @property
    def name(self) -> str:
        return "arXiv"
    
    def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any
    ) -> List[PaperMetadata]:
        """Search arXiv for papers."""
        search_query = self._build_query(query, **kwargs)
        sort_by = self._get_sort_criterion(kwargs.get("sort_by", "relevance"))
        
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=arxiv.SortOrder.Descending,
        )
        
        papers = []
        for result in self.client.results(search):
            paper = self._convert_to_metadata(result)
            if self._passes_filters(paper, **kwargs):
                papers.append(paper)
        
        return papers[:max_results]
    
    def get_paper(self, paper_id: str) -> Optional[PaperMetadata]:
        """Get paper by arXiv ID."""
        # Clean ID (remove 'arXiv:' prefix if present)
        clean_id = paper_id.replace("arXiv:", "").strip()
        
        search = arxiv.Search(id_list=[clean_id])
        results = list(self.client.results(search))
        
        if results:
            return self._convert_to_metadata(results[0])
        return None
    
    def download_pdf(self, paper: PaperMetadata, output_path: str) -> bool:
        """Download PDF from arXiv."""
        if not paper.arxiv_id:
            return False
        
        try:
            search = arxiv.Search(id_list=[paper.arxiv_id])
            result = next(self.client.results(search))
            result.download_pdf(filename=output_path)
            return True
        except Exception as e:
            print(f"arXiv download error: {e}")
            return False
    
    def _build_query(self, query: str, **kwargs: Any) -> str:
        """Build arXiv query string."""
        parts = [query]
        
        if kwargs.get("categories"):
            cats = kwargs["categories"]
            cat_query = " OR ".join([f"cat:{c}" for c in cats])
            parts.append(f"({cat_query})")
        
        if kwargs.get("author"):
            parts.append(f'au:{kwargs["author"]}')
        
        return " AND ".join(parts)
    
    def _get_sort_criterion(self, sort_by: str) -> arxiv.SortCriterion:
        """Convert sort string to arxiv criterion."""
        mapping = {
            "relevance": arxiv.SortCriterion.Relevance,
            "date": arxiv.SortCriterion.LastUpdatedDate,
            "submitted": arxiv.SortCriterion.SubmittedDate,
        }
        return mapping.get(sort_by, arxiv.SortCriterion.Relevance)
    
    def _convert_to_metadata(self, result: arxiv.Result) -> PaperMetadata:
        """Convert arxiv.Result to PaperMetadata."""
        authors = [
            Author(name=a.name) for a in result.authors
        ]
        
        return PaperMetadata(
            title=result.title,
            authors=authors,
            year=result.published.year if result.published else None,
            doi=result.doi,
            arxiv_id=result.get_short_id(),
            source=SourceType.ARXIV,
            url=result.entry_id,
            pdf_url=result.pdf_url,
            abstract=result.summary,
            categories=list(result.categories),
            journal=result.journal_ref,
            published_date=result.published,
        )
    
    def _passes_filters(self, paper: PaperMetadata, **kwargs: Any) -> bool:
        """Check if paper passes year filters."""
        if kwargs.get("year_from") and paper.year:
            if paper.year < kwargs["year_from"]:
                return False
        if kwargs.get("year_to") and paper.year:
            if paper.year > kwargs["year_to"]:
                return False
        return True
