"""
Configuration and constants for paper downloader.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Config:
    """Application configuration."""
    
    # Search settings
    max_results: int = 50
    download_delay: int = 2
    
    # NCBI/PubMed credentials - Use environment variables in production
    ncbi_api_key: str = field(
        default_factory=lambda: os.getenv("NCBI_API_KEY", "")
    )
    ncbi_email: str = field(
        default_factory=lambda: os.getenv("NCBI_EMAIL", "")
    )
    
    # Folder structure
    pdf_folder: str = "papers_pdf"
    markdown_folder: str = "papers_markdown"
    metadata_file: str = "papers_metadata.json"


# arXiv Categories for filtering
ARXIV_CATEGORIES: Dict[str, str] = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.CV": "Computer Vision",
    "cs.CL": "Computation and Language",
    "cs.NE": "Neural and Evolutionary Computing",
    "q-bio.GN": "Genomics",
    "q-bio.QM": "Quantitative Methods",
    "physics.bio-ph": "Biological Physics",
    "stat.ML": "Machine Learning (Statistics)",
    "math.ST": "Statistics Theory",
}


def get_config() -> Config:
    """Get configuration instance."""
    return Config()
