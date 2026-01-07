"""
Paper Downloader - Modular paper download and extraction system.
"""
from paper_downloader.config import Config, get_config
from paper_downloader.database import PaperDatabase
from paper_downloader.extractor import MarkerExtractor

__version__ = "1.0.0"
__all__ = ["Config", "get_config", "PaperDatabase", "MarkerExtractor"]
