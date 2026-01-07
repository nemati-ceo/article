"""
Abstract base class for paper providers.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from paper_downloader.database import PaperDatabase


class BaseProvider(ABC):
    """Abstract base class for paper providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @abstractmethod
    def search(
        self, query: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """Search for papers."""
        pass

    @abstractmethod
    def extract_metadata(self, paper: Any) -> Dict[str, Any]:
        """Extract metadata from paper object."""
        pass

    @abstractmethod
    def download_pdf(
        self, paper: Any, pdf_path: str
    ) -> bool:
        """Download PDF for a paper. Returns True if successful."""
        pass

    def download_and_save(
        self,
        paper: Any,
        pdf_folder: str,
        db: PaperDatabase,
    ) -> Tuple[bool, Optional[str]]:
        """
        Download PDF and save metadata to database.
        Returns (success, paper_uuid).
        """
        import os
        from paper_downloader.utils import sanitize_filename

        os.makedirs(pdf_folder, exist_ok=True)
        metadata = self.extract_metadata(paper)

        safe_name = sanitize_filename(metadata["title"])
        identifier = metadata.get("arxiv_id") or metadata.get("pmc_id", "unknown")
        pdf_filename = f"{identifier}_{safe_name}.pdf"
        pdf_path = os.path.join(pdf_folder, pdf_filename)

        print(f"   ⬇️ Downloading: {metadata['title'][:50]}...")

        if self.download_pdf(paper, pdf_path):
            print(f"   ✅ PDF saved: {pdf_path}")
            metadata["pdf_path"] = pdf_path

            paper_uuid, citation = db.add_paper(metadata)
            print(f"   🆔 UUID: {paper_uuid}")
            print(f"   📝 Citation: {citation[:80]}...")

            return True, paper_uuid

        print("   ❌ Download failed")
        return False, None
