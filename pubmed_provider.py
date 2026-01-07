"""
PubMed Central paper provider implementation.
"""
import io
import re
import tarfile
from typing import Any, Dict, List, Optional

import requests
from Bio import Entrez

from .base import BaseProvider


class PubMedProvider(BaseProvider):
    """Provider for PubMed Central papers."""

    def __init__(self, email: str, api_key: str, max_results: int = 50):
        self.email = email
        self.api_key = api_key
        self.max_results = max_results
        Entrez.email = email
        Entrez.api_key = api_key

    @property
    def name(self) -> str:
        return "PubMed Central"

    def search(
        self, query: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search PubMed Central for papers."""
        try:
            handle = Entrez.esearch(db="pmc", term=query, retmax=self.max_results)
            record = Entrez.read(handle)
            handle.close()

            ids = record.get("IdList", [])
            if not ids:
                return []

            handle = Entrez.esummary(db="pmc", id=",".join(ids))
            summaries = Entrez.read(handle)
            handle.close()

            papers = []
            for summary in summaries:
                papers.append({
                    "pmc_id": f"PMC{summary.get('Id', '')}",
                    "title": summary.get("Title", "No title"),
                    "authors": summary.get("AuthorList", []),
                    "date": summary.get("PubDate", "N/A"),
                    "source": summary.get("Source", ""),
                })

            return papers

        except Exception as e:
            print(f"   ❌ PubMed API Error: {e}")
            return []

    def extract_metadata(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from PubMed paper dict."""
        authors = paper.get("authors", [])
        if isinstance(authors, list):
            authors = authors[:3] if authors else ["Unknown"]

        return {
            "title": paper["title"],
            "authors": authors,
            "year": str(paper["date"])[:4] if paper["date"] else "N/A",
            "source": "PubMed Central",
            "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{paper['pmc_id']}/",
            "pmc_id": paper["pmc_id"],
            "abstract": self._get_abstract(paper["pmc_id"]),
        }

    def _get_abstract(self, pmc_id: str) -> str:
        """Fetch abstract for a single PMC article."""
        try:
            handle = Entrez.efetch(
                db="pmc", id=pmc_id.replace("PMC", ""), rettype="xml"
            )
            content = handle.read()
            handle.close()

            if b"<abstract>" in content:
                start = content.find(b"<abstract>") + 10
                end = content.find(b"</abstract>")
                abstract = content[start:end].decode("utf-8", errors="ignore")
                abstract = re.sub(r"<[^>]+>", "", abstract).strip()
                return abstract

        except Exception:
            pass

        return "Abstract not available"

    def download_pdf(self, paper: Dict[str, Any], pdf_path: str) -> bool:
        """Download PDF from PubMed Central."""
        pmc_id = paper["pmc_id"]
        oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmc_id}"

        try:
            response = requests.get(oa_url, timeout=30)
            if response.status_code != 200:
                return False

            tgz_match = re.search(r'href="(ftp://[^"]+\.tar\.gz)"', response.text)
            if not tgz_match:
                return False

            tgz_url = tgz_match.group(1)
            tgz_url = tgz_url.replace(
                "ftp://ftp.ncbi.nlm.nih.gov/", "https://ftp.ncbi.nlm.nih.gov/"
            )

            response = requests.get(
                tgz_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=120
            )
            if response.status_code != 200:
                return False

            tar_bytes = io.BytesIO(response.content)
            with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".pdf"):
                        pdf_file = tar.extractfile(member)
                        if pdf_file:
                            with open(pdf_path, "wb") as f:
                                f.write(pdf_file.read())
                            return True

            return False

        except Exception as e:
            print(f"      Error: {e}")
            return False
