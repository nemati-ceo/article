"""
Marker AI processor for PDF to structured text extraction.
"""
import re
from typing import Dict, Optional

from paperflow.schemas import Section, SectionType


class MarkerProcessor:
    """PDF processor using Marker AI."""

    def __init__(self):
        self._converter = None
        self._text_from_rendered = None
        self.available = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize Marker AI models."""
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered

            print("⏳ Loading Marker AI models...")
            self._converter = PdfConverter(artifact_dict=create_model_dict())
            self._text_from_rendered = text_from_rendered
            self.available = True
            print("✅ Marker AI loaded")

        except ImportError as e:
            print(f"⚠️ Marker AI not available: {e}")
            print("   Install with: pip install paperflow[extraction]")
            self.available = False
        except Exception as e:
            print(f"❌ Marker AI error: {e}")
            self.available = False

    def extract_full_text(self, pdf_path: str) -> Optional[str]:
        """Extract full text from PDF."""
        if not self.available:
            return None

        try:
            rendered = self._converter(pdf_path)
            full_text, _, _ = self._text_from_rendered(rendered)
            return full_text
        except Exception as e:
            print(f"Extraction error: {e}")
            return None

    def extract_sections(self, pdf_path: str) -> Dict[SectionType, Section]:
        """Extract structured sections from PDF."""
        full_text = self.extract_full_text(pdf_path)
        if not full_text:
            return {}

        return self.parse_sections(full_text)

    def parse_sections(self, text: str) -> Dict[SectionType, Section]:
        """Parse sections from markdown text."""
        sections = {}

        patterns = {
            SectionType.ABSTRACT: (
                r"abstract",
                r"introduction|keywords|1\s+introduction|1\.\s+introduction"
            ),
            SectionType.INTRODUCTION: (
                r"introduction|1\s+introduction|1\.\s+introduction",
                r"related\s+work|background|methodology|method|2\s+|2\.\s+"
            ),
            SectionType.BACKGROUND: (
                r"background|related\s+work|literature\s+review",
                r"methodology|method|approach|3\s+|3\.\s+"
            ),
            SectionType.METHODS: (
                r"method|methodology|approach|experimental\s+setup",
                r"result|experiment|evaluation|4\s+|4\.\s+"
            ),
            SectionType.RESULTS: (
                r"result|experiment|evaluation|finding",
                r"discussion|conclusion|5\s+|5\.\s+|6\s+"
            ),
            SectionType.DISCUSSION: (
                r"discussion",
                r"conclusion|limitation|future\s+work|reference"
            ),
            SectionType.CONCLUSION: (
                r"conclusion|summary|concluding\s+remark",
                r"reference|acknowledgment|appendix|bibliography"
            ),
        }

        for section_type, (start_pattern, end_pattern) in patterns.items():
            content = self._extract_section(text, start_pattern, end_pattern)
            if content and len(content.strip()) > 50:
                sections[section_type] = Section(
                    section_type=section_type,
                    content=self._clean_section(content),
                    order=list(SectionType).index(section_type)
                )

        sections[SectionType.FULL_TEXT] = Section(
            section_type=SectionType.FULL_TEXT,
            content=text,
            order=99
        )

        return sections

    def _extract_section(self, text: str, start_pattern: str, end_pattern: str) -> str:
        """Extract section between patterns."""
        start_match = re.search(
            r"(?:^|\n)#*\s*" + start_pattern,
            text,
            re.IGNORECASE | re.MULTILINE
        )
        if not start_match:
            return ""

        start_pos = start_match.end()

        newline_pos = text.find("\n", start_pos)
        if newline_pos != -1:
            start_pos = newline_pos + 1

        end_match = re.search(
            r"(?:^|\n)#*\s*" + end_pattern,
            text[start_pos:],
            re.IGNORECASE | re.MULTILINE
        )

        if end_match:
            end_pos = start_pos + end_match.start()
            return text[start_pos:end_pos]

        return text[start_pos:start_pos + 5000]

    def _clean_section(self, content: str) -> str:
        """Clean extracted section content."""
        content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)
        content = re.sub(r"^\s*#+ ", "", content, flags=re.MULTILINE)
        return content.strip()


class SectionExtractor:
    """Lightweight section extractor."""

    def __init__(self):
        self.marker = MarkerProcessor()

    def extract_from_pdf(self, pdf_path: str) -> Dict[SectionType, Section]:
        """Extract sections from PDF file."""
        return self.marker.extract_sections(pdf_path)

    def extract_from_text(self, text: str) -> Dict[SectionType, Section]:
        """Extract sections from text."""
        return self.marker.parse_sections(text)

    def extract_abstract_from_metadata(self, abstract: str) -> Optional[Section]:
        """Create Section from metadata abstract."""
        if not abstract:
            return None

        return Section(
            section_type=SectionType.ABSTRACT,
            content=abstract,
            order=0
        )
