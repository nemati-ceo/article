"""
Paper metadata database with UUID and citation management.
"""
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


class PaperDatabase:
    """Manages paper metadata with UUIDs and citations."""

    def __init__(self, db_file: str = "papers_metadata.json"):
        self.db_file = db_file
        self.papers: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        """Load existing metadata database."""
        if os.path.exists(self.db_file):
            with open(self.db_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save(self) -> None:
        """Save metadata database."""
        with open(self.db_file, "w", encoding="utf-8") as f:
            json.dump(self.papers, f, indent=2, ensure_ascii=False)

    def add_paper(self, metadata: Dict[str, Any]) -> Tuple[str, str]:
        """Add a paper with UUID and citation. Returns (uuid, citation)."""
        paper_uuid = str(uuid.uuid4())
        citation = self._generate_citation(metadata)

        self.papers[paper_uuid] = {
            "uuid": paper_uuid,
            "title": metadata["title"],
            "authors": metadata["authors"],
            "year": metadata.get("year", "N/A"),
            "source": metadata["source"],
            "arxiv_id": metadata.get("arxiv_id", "N/A"),
            "doi": metadata.get("doi", "N/A"),
            "url": metadata["url"],
            "pdf_path": metadata.get("pdf_path", "N/A"),
            "markdown_path": metadata.get("markdown_path", "N/A"),
            "citation": citation,
            "categories": metadata.get("categories", []),
            "abstract": metadata.get("abstract", ""),
            "added_date": datetime.now().isoformat(),
        }

        self.save()
        return paper_uuid, citation

    def _generate_citation(self, metadata: Dict[str, Any]) -> str:
        """Generate APA-style citation."""
        authors = metadata["authors"]
        author_str = self._format_authors(authors)
        year = metadata.get("year", "n.d.")
        title = metadata["title"]

        if metadata["source"] == "arXiv":
            arxiv_id = metadata.get("arxiv_id", "")
            return f"{author_str} ({year}). {title}. arXiv preprint arXiv:{arxiv_id}."
        return f"{author_str} ({year}). {title}. {metadata['source']}."

    def _format_authors(self, authors: Any) -> str:
        """Format author list for citation."""
        if isinstance(authors, list):
            if len(authors) == 1:
                return authors[0]
            elif len(authors) == 2:
                return f"{authors[0]} & {authors[1]}"
            elif len(authors) > 2:
                return f"{authors[0]} et al."
        return str(authors) if authors else "Unknown"

    def get_paper(self, paper_uuid: str) -> Optional[Dict[str, Any]]:
        """Retrieve paper by UUID."""
        return self.papers.get(paper_uuid)

    def list_papers(self) -> Dict[str, Dict[str, Any]]:
        """List all papers with UUIDs."""
        return self.papers

    def update_paper(self, paper_uuid: str, updates: Dict[str, Any]) -> None:
        """Update paper metadata."""
        if paper_uuid in self.papers:
            self.papers[paper_uuid].update(updates)
            self.save()

    def __len__(self) -> int:
        return len(self.papers)
