"""
PDF extraction using Marker AI.
"""
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional

from paper_downloader.utils import sanitize_filename


class MarkerExtractor:
    """PDF to Markdown extractor using Marker AI."""

    def __init__(self):
        self.converter = None
        self.available = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize Marker AI models."""
        print("⏳ Loading Marker AI models...")
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered

            self.converter = PdfConverter(artifact_dict=create_model_dict())
            self._text_from_rendered = text_from_rendered
            self.available = True
            print("✅ Marker models loaded.")

        except ImportError as e:
            print(f"❌ Marker import error: {e}")
            print("   Install with: pip install marker-pdf --upgrade")
            self.available = False

        except Exception as e:
            print(f"❌ Marker load error: {e}")
            self.available = False

    def extract_sections(self, pdf_path: str) -> Optional[Dict[str, str]]:
        """Extract important sections: Abstract, Introduction, Conclusion."""
        if not self.available:
            print("   ⚠️ Marker AI not available. Skipping extraction.")
            return None

        try:
            print("   🤖 Processing with Marker AI...")

            rendered = self.converter(pdf_path)
            full_text, _, _ = self._text_from_rendered(rendered)

            print(f"   📝 Extracted {len(full_text)} characters")

            sections = {
                "abstract": self._extract_section(
                    full_text,
                    r"abstract",
                    r"introduction|keywords|1\s+introduction",
                ),
                "introduction": self._extract_section(
                    full_text,
                    r"introduction|1\s+introduction",
                    r"related work|methodology|method|background|2\s+",
                ),
                "conclusion": self._extract_section(
                    full_text,
                    r"conclusion|conclusions|discussion and conclusion",
                    r"references|acknowledgment|appendix|bibliography",
                ),
            }

            # Clean up sections
            for key in sections:
                if sections[key]:
                    sections[key] = re.sub(r"\n\s*\n\s*\n+", "\n\n", sections[key])
                    sections[key] = sections[key].strip()

            return sections

        except Exception as e:
            print(f"   ⚠️ Marker AI extraction failed: {e}")
            return None

    def _extract_section(
        self, text: str, start_pattern: str, end_pattern: str
    ) -> str:
        """Extract text between section headers."""
        start_match = re.search(r"\b" + start_pattern + r"\b", text, re.IGNORECASE)
        if not start_match:
            start_match = re.search(start_pattern, text, re.IGNORECASE)
            if not start_match:
                return ""

        start_pos = start_match.end()

        # Skip header line, find first substantial paragraph
        lines = text[start_pos:].split("\n")
        actual_start = 0
        for i, line in enumerate(lines):
            if line.strip() and len(line.strip()) > 20:
                actual_start = sum(len(ln) + 1 for ln in lines[:i])
                break

        start_pos += actual_start

        # Find end of section
        end_match = re.search(r"\b" + end_pattern + r"\b", text[start_pos:], re.IGNORECASE)
        if not end_match:
            end_match = re.search(end_pattern, text[start_pos:], re.IGNORECASE)

        if end_match:
            end_pos = start_pos + end_match.start()
            return text[start_pos:end_pos].strip()

        # Take next 3000 chars if no end found
        return text[start_pos : start_pos + 3000].strip()

    def create_markdown(
        self,
        paper_uuid: str,
        metadata: Dict[str, Any],
        sections: Dict[str, str],
        output_folder: str,
    ) -> str:
        """Create markdown file with extracted sections."""
        os.makedirs(output_folder, exist_ok=True)

        safe_name = sanitize_filename(metadata["title"])
        md_filename = f"{paper_uuid[:8]}_{safe_name}.md"
        md_path = os.path.join(output_folder, md_filename)

        authors = metadata["authors"]
        if isinstance(authors, list):
            authors = ", ".join(authors)

        categories = metadata.get("categories", [])
        if isinstance(categories, list):
            categories = ", ".join(categories) if categories else "N/A"

        content = self._build_markdown_content(
            paper_uuid, metadata, sections, authors, categories
        )

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(content)

        return md_path

    def _build_markdown_content(
        self,
        paper_uuid: str,
        metadata: Dict[str, Any],
        sections: Dict[str, str],
        authors: str,
        categories: str,
    ) -> str:
        """Build markdown content string."""
        abstract = sections.get("abstract") or metadata.get("abstract", "Not available")
        intro = sections.get("introduction", "*Section not extracted*")
        conclusion = sections.get("conclusion", "*Section not extracted*")

        return f"""# {metadata['title']}

**UUID:** `{paper_uuid}`

---

## 📋 Metadata

| Field | Value |
|-------|-------|
| **Authors** | {authors} |
| **Year** | {metadata.get('year', 'N/A')} |
| **Source** | {metadata['source']} |
| **arXiv ID** | {metadata.get('arxiv_id', 'N/A')} |
| **DOI** | {metadata.get('doi', 'N/A')} |
| **Categories** | {categories} |

**🔗 URL:** [{metadata['url']}]({metadata['url']})

---

## 📖 Citation

```
{metadata['citation']}
```

---

## 📝 Abstract

{abstract}

---

## 🔍 Introduction

{intro}

---

## 🎯 Conclusion

{conclusion}

---

## 📎 File Information

- **PDF Location:** `{metadata.get('pdf_path', 'N/A')}`
- **Extracted:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Extraction Method:** Marker AI

---

*Key sections only. See full PDF for complete content.*
"""
