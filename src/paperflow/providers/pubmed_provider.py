"""
PubMed/PMC provider implementation.
"""
import io
import os
import re
import tarfile
from typing import Any, List, Optional

import requests
from Bio import Entrez

from paperflow.schemas import Author, PaperMetadata, SourceType
from .base import BaseProvider


class PubMedProvider(BaseProvider):
    """Provider for PubMed/PMC papers."""

    def __init__(
        self,
        email: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.email = email or os.getenv("NCBI_EMAIL", "")
        self.api_key = api_key or os.getenv("NCBI_API_KEY", "")

        Entrez.email = self.email
        if self.api_key:
            Entrez.api_key = self.api_key

    @property
    def source_type(self) -> SourceType:
        return SourceType.PUBMED

    @property
    def name(self) -> str:
        return "PubMed"

    def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any
    ) -> List[PaperMetadata]:
        """Search PubMed for papers."""
        search_query = self._build_query(query, **kwargs)

        try:
            handle = Entrez.esearch(db="pmc", term=search_query, retmax=max_results)
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
                paper = self._convert_to_metadata(summary)
                papers.append(paper)

            return papers

        except Exception as e:
            print(f"PubMed search error: {e}")
            return []

    def get_paper(self, paper_id: str) -> Optional[PaperMetadata]:
        """Get paper by PMID or PMC ID."""
        if paper_id.upper().startswith("PMC"):
            db = "pmc"
            clean_id = paper_id.upper().replace("PMC", "")
        else:
            db = "pubmed"
            clean_id = paper_id

        try:
            handle = Entrez.esummary(db=db, id=clean_id)
            summaries = Entrez.read(handle)
            handle.close()

            if summaries:
                return self._convert_to_metadata(summaries[0], db=db)
            return None

        except Exception as e:
            print(f"PubMed get_paper error: {e}")
            return None

    def download_pdf(self, paper: PaperMetadata, output_path: str) -> bool:
        """Download PDF from PubMed Central."""
        pmc_id = paper.pmc_id
        if not pmc_id:
            return False

        oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmc_id}"

        try:
            response = requests.get(oa_url, timeout=30)
            if response.status_code != 200:
                return False

            tgz_match = re.search(r'href="(ftp://[^"]+\.tar\.gz)"', response.text)
            if not tgz_match:
                return False

            tgz_url = tgz_match.group(1).replace(
                "ftp://ftp.ncbi.nlm.nih.gov/",
                "https://ftp.ncbi.nlm.nih.gov/"
            )

            response = requests.get(
                tgz_url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=120
            )
            if response.status_code != 200:
                return False

            tar_bytes = io.BytesIO(response.content)
            with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".pdf"):
                        pdf_file = tar.extractfile(member)
                        if pdf_file:
                            with open(output_path, "wb") as f:
                                f.write(pdf_file.read())
                            return True

            return False

        except Exception as e:
            print(f"PubMed download error: {e}")
            return False

    def get_abstract(self, pmc_id: str) -> str:
        """Fetch abstract for a PMC article."""
        try:
            clean_id = pmc_id.upper().replace("PMC", "")
            handle = Entrez.efetch(db="pmc", id=clean_id, rettype="xml")
            content = handle.read()
            handle.close()

            if b"<abstract>" in content:
                start = content.find(b"<abstract>") + 10
                end = content.find(b"</abstract>")
                abstract = content[start:end].decode("utf-8", errors="ignore")
                return re.sub(r"<[^>]+>", "", abstract).strip()
        except Exception:
            pass

        return ""

    def _build_query(self, query: str, **kwargs: Any) -> str:
        """Build PubMed query string."""
        parts = [query]

        if kwargs.get("year_from"):
            parts.append(f'{kwargs["year_from"]}[PDAT]')
        if kwargs.get("year_to"):
            parts.append(f'{kwargs["year_to"]}[PDAT]')
        if kwargs.get("author"):
            parts.append(f'{kwargs["author"]}[AUTH]')

        return " AND ".join(parts)

    def _convert_to_metadata(self, summary: dict, db: str = "pmc") -> PaperMetadata:
        """Convert Entrez summary to PaperMetadata."""
        pmc_id = f"PMC{summary.get('Id', '')}" if db == "pmc" else None
        pmid = summary.get("Id") if db == "pubmed" else None

        pub_date = summary.get("PubDate", "")
        year = None
        if pub_date:
            match = re.search(r"(\d{4})", str(pub_date))
            if match:
                year = int(match.group(1))

        author_list = summary.get("AuthorList", [])
        authors = [Author(name=a) for a in author_list] if author_list else []

        return PaperMetadata(
            title=summary.get("Title", ""),
            authors=authors,
            year=year,
            doi=summary.get("DOI"),
            pmid=pmid,
            pmc_id=pmc_id,
            source=SourceType.PUBMED,
            url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/" if pmc_id else "",
            abstract=self.get_abstract(pmc_id) if pmc_id else "",
            journal=summary.get("Source", ""),
            volume=summary.get("Volume"),
            issue=summary.get("Issue"),
            pages=summary.get("Pages"),
        )
