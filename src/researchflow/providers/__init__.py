"""
Paper providers package.
Unified interface to multiple academic paper sources.
"""
from typing import Dict, List, Optional, Type

from paperflow.schemas import PaperMetadata, SearchQuery, SearchResult, SourceType

from .base import BaseProvider
from .arxiv_provider import ArxivProvider
from .pubmed_provider import PubMedProvider
from .semantic_scholar_provider import SemanticScholarProvider
from .openalex_provider import OpenAlexProvider


PROVIDER_REGISTRY: Dict[SourceType, Type[BaseProvider]] = {
    SourceType.ARXIV: ArxivProvider,
    SourceType.PUBMED: PubMedProvider,
    SourceType.SEMANTIC_SCHOLAR: SemanticScholarProvider,
    SourceType.OPENALEX: OpenAlexProvider,
}


def get_provider(source: SourceType, **kwargs) -> BaseProvider:
    """Get a provider instance by source type."""
    if source not in PROVIDER_REGISTRY:
        raise ValueError(f"Unknown source: {source}")
    provider_class = PROVIDER_REGISTRY[source]
    return provider_class(**kwargs)


def get_all_providers(**kwargs) -> Dict[SourceType, BaseProvider]:
    """Get instances of all available providers."""
    return {
        source: get_provider(source, **kwargs)
        for source in PROVIDER_REGISTRY
    }


class UnifiedSearch:
    """Unified search across multiple providers."""

    def __init__(self, **provider_kwargs):
        self._providers: Dict[SourceType, BaseProvider] = {}
        self._provider_kwargs = provider_kwargs

    def _get_provider(self, source: SourceType) -> BaseProvider:
        """Get or create provider instance."""
        if source not in self._providers:
            self._providers[source] = get_provider(source, **self._provider_kwargs)
        return self._providers[source]

    def search(
        self,
        query: str,
        sources: Optional[List[SourceType]] = None,
        max_results: int = 10,
        **kwargs
    ) -> SearchResult:
        """Search across multiple sources."""
        import time
        start_time = time.time()

        if sources is None:
            sources = list(PROVIDER_REGISTRY.keys())

        all_papers: List[PaperMetadata] = []
        searched_sources: List[SourceType] = []

        for source in sources:
            try:
                provider = self._get_provider(source)
                papers = provider.search(
                    query=query,
                    max_results=max_results,
                    **kwargs
                )
                all_papers.extend(papers)
                searched_sources.append(source)
            except Exception as e:
                print(f"Error searching {source.value}: {e}")

        # Deduplicate by DOI
        seen_dois = set()
        unique_papers = []
        for paper in all_papers:
            if paper.doi:
                if paper.doi in seen_dois:
                    continue
                seen_dois.add(paper.doi)
            unique_papers.append(paper)

        elapsed_ms = int((time.time() - start_time) * 1000)

        return SearchResult(
            query=SearchQuery(
                query=query,
                sources=sources,
                max_results=max_results,
                **{k: v for k, v in kwargs.items() if k in SearchQuery.model_fields}
            ),
            papers=unique_papers,
            total_found=len(unique_papers),
            sources_searched=searched_sources,
            search_time_ms=elapsed_ms
        )

    def get_paper(
        self,
        paper_id: str,
        source: Optional[SourceType] = None
    ) -> Optional[PaperMetadata]:
        """Get a paper by ID."""
        if source is None:
            source = self._detect_source(paper_id)

        if source:
            provider = self._get_provider(source)
            return provider.get_paper(paper_id)

        for src in PROVIDER_REGISTRY:
            try:
                provider = self._get_provider(src)
                paper = provider.get_paper(paper_id)
                if paper:
                    return paper
            except Exception:
                continue

        return None

    def _detect_source(self, paper_id: str) -> Optional[SourceType]:
        """Detect source type from paper ID format."""
        paper_id_lower = paper_id.lower()

        if paper_id_lower.startswith("arxiv:") or "." in paper_id and paper_id[0].isdigit():
            return SourceType.ARXIV
        elif paper_id_lower.startswith("pmc") or paper_id.isdigit():
            return SourceType.PUBMED
        elif paper_id.startswith("10."):
            return SourceType.OPENALEX
        elif paper_id.startswith("W") and paper_id[1:].isdigit():
            return SourceType.OPENALEX

        return None


__all__ = [
    "BaseProvider",
    "ArxivProvider",
    "PubMedProvider",
    "SemanticScholarProvider",
    "OpenAlexProvider",
    "UnifiedSearch",
    "get_provider",
    "get_all_providers",
    "PROVIDER_REGISTRY",
]
