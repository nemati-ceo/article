# paperflow/__init__.py

from importlib.metadata import version as _version
import os
import sys

__version__ = _version("paperflow")


def _show_banner() -> None:
    banner = f"""
╭───────────────────────────────────────────────────────────────╮
│                    paperflow                                  │
│            Academic paper ingestion                           │
│                                                               │
│  search → download → extract Markditdown                      │
│                                                               │
│  version: {__version__:<28}                                   │
╰───────────────────────────────────────────────────────────────╯
""".rstrip()

    print(banner)


# Show banner only in interactive usage
if (
    os.environ.get("PAPERFLOW_NO_BANNER") != "1"
    and hasattr(sys, "ps1")  # interactive shell
):
    _show_banner()

from paperflow.pipeline import PaperPipeline
from paperflow.schemas import (
    Paper,
    PaperMetadata,
    SearchResult,
    Section,
    Chunk,
    SourceType,
    SectionType,
)
from paperflow.providers import UnifiedSearch

__all__ = [
    "PaperPipeline",
    "Paper",
    "PaperMetadata",
    "SearchResult",
    "Section",
    "Chunk",
    "SourceType",
    "SectionType",
    "UnifiedSearch",
]
