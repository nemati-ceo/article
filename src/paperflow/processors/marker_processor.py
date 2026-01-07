"""
PDF to structured text extraction.
Supports: Marker AI (high quality), MarkItDown (lightweight), Docling (IBM).
"""
import re
from enum import Enum
from typing import Dict, Optional

from paperflow.schemas import Section, SectionType


class ExtractorBackend(str, Enum):
    """Available extraction backends."""
    MARKER = "marker"
    MARKITDOWN = "markitdown"
    DOCLING = "docling"
    AUTO = "auto"


class MarkerProcessor:
    """PDF processor using Marker AI."""

    def __init__(self, gpu: bool = False):
        self.gpu = gpu
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
            
            device = "cuda" if self.gpu else "cpu"
            if self.gpu:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        print("⚠️ GPU requested but CUDA not available, falling back to CPU")
                        device = "cpu"
                    else:
                        print(f"🎯 Using GPU for Marker AI ({torch.cuda.get_device_name()})")
                except ImportError:
                    print("⚠️ PyTorch not available for GPU check, using CPU")
                    device = "cpu"
            
            config = {"torch_device": device}
            self._converter = PdfConverter(
                artifact_dict=create_model_dict(device=device),
                config=config
            )
            self._text_from_rendered = text_from_rendered
            self.available = True
            print("✅ Marker AI loaded")

        except ImportError as e:
            print(f"⚠️ Marker AI not available: {e}")
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
            print(f"Marker extraction error: {e}")
            return None


class MarkItDownProcessor:
    """PDF processor using Microsoft MarkItDown (lightweight)."""

    def __init__(self):
        self._converter = None
        self.available = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize MarkItDown."""
        try:
            from markitdown import MarkItDown
            
            self._converter = MarkItDown()
            self.available = True
            print("✅ MarkItDown loaded")

        except ImportError as e:
            print(f"⚠️ MarkItDown not available: {e}")
            self.available = False
        except Exception as e:
            print(f"❌ MarkItDown error: {e}")
            self.available = False

    def extract_full_text(self, pdf_path: str) -> Optional[str]:
        """Extract full text from PDF."""
        if not self.available:
            return None

        try:
            result = self._converter.convert(pdf_path)
            return result.text_content
        except Exception as e:
            print(f"MarkItDown extraction error: {e}")
            return None


class DoclingProcessor:
    """PDF processor using IBM Docling."""

    def __init__(self, gpu: bool = False):
        self.gpu = gpu
        self._converter = None
        self.available = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize Docling."""
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import PdfFormatOption
            
            print("⏳ Loading Docling...")
            
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            
            if self.gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        pipeline_options.accelerator_options = {"device": "cuda"}
                        print(f"🎯 Using GPU for Docling ({torch.cuda.get_device_name()})")
                    else:
                        print("⚠️ GPU requested but CUDA not available, using CPU")
                except ImportError:
                    print("⚠️ PyTorch not available for GPU check, using CPU")
            
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            self.available = True
            print("✅ Docling loaded")

        except ImportError as e:
            print(f"⚠️ Docling not available: {e}")
            print("   Install with: pip install docling")
            self.available = False
        except Exception as e:
            print(f"❌ Docling error: {e}")
            self.available = False

    def extract_full_text(self, pdf_path: str) -> Optional[str]:
        """Extract full text from PDF."""
        if not self.available:
            return None

        try:
            result = self._converter.convert(pdf_path)
            return result.document.export_to_markdown()
        except Exception as e:
            print(f"Docling extraction error: {e}")
            return None

    def extract_structured(self, pdf_path: str) -> Optional[Dict]:
        """Extract structured content with tables and figures."""
        if not self.available:
            return None

        try:
            result = self._converter.convert(pdf_path)
            doc = result.document
            
            return {
                "markdown": doc.export_to_markdown(),
                "text": doc.export_to_text(),
                "tables": [table.export_to_dataframe() for table in doc.tables],
                "figures": [fig.export() for fig in doc.figures] if hasattr(doc, 'figures') else [],
            }
        except Exception as e:
            print(f"Docling structured extraction error: {e}")
            return None


class PDFExtractor:
    """
    Unified PDF extractor with multiple backend support.
    
    Backends:
        - marker: High quality, GPU support, best for academic papers
        - markitdown: Lightweight, fast, CPU only, Microsoft
        - docling: IBM, good table extraction, GPU support
        - auto: Try marker → docling → markitdown
    """

    def __init__(
        self,
        backend: str | ExtractorBackend = ExtractorBackend.AUTO,
        gpu: bool = False
    ):
        self.backend = ExtractorBackend(backend) if isinstance(backend, str) else backend
        self.gpu = gpu
        
        self._marker: Optional[MarkerProcessor] = None
        self._markitdown: Optional[MarkItDownProcessor] = None
        self._docling: Optional[DoclingProcessor] = None
        
        self._active_backend: Optional[str] = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the selected backend(s)."""
        if self.backend == ExtractorBackend.MARKER:
            self._marker = MarkerProcessor(gpu=self.gpu)
            if self._marker.available:
                self._active_backend = "marker"
                
        elif self.backend == ExtractorBackend.MARKITDOWN:
            self._markitdown = MarkItDownProcessor()
            if self._markitdown.available:
                self._active_backend = "markitdown"
                
        elif self.backend == ExtractorBackend.DOCLING:
            self._docling = DoclingProcessor(gpu=self.gpu)
            if self._docling.available:
                self._active_backend = "docling"
                
        elif self.backend == ExtractorBackend.AUTO:
            # Try in order: marker → docling → markitdown
            self._marker = MarkerProcessor(gpu=self.gpu)
            if self._marker.available:
                self._active_backend = "marker"
            else:
                self._docling = DoclingProcessor(gpu=self.gpu)
                if self._docling.available:
                    self._active_backend = "docling"
                else:
                    self._markitdown = MarkItDownProcessor()
                    if self._markitdown.available:
                        self._active_backend = "markitdown"

    @property
    def available(self) -> bool:
        """Check if any backend is available."""
        return self._active_backend is not None

    @property
    def active_backend(self) -> Optional[str]:
        """Get the active backend name."""
        return self._active_backend

    def extract_full_text(self, pdf_path: str) -> Optional[str]:
        """Extract full text from PDF using active backend."""
        if self._active_backend == "marker" and self._marker:
            return self._marker.extract_full_text(pdf_path)
        elif self._active_backend == "docling" and self._docling:
            return self._docling.extract_full_text(pdf_path)
        elif self._active_backend == "markitdown" and self._markitdown:
            return self._markitdown.extract_full_text(pdf_path)
        return None

    def extract_sections(self, pdf_path: str) -> Dict[str, dict]:
        """Extract structured sections from PDF."""
        full_text = self.extract_full_text(pdf_path)
        if not full_text:
            return {}

        return self.parse_sections(full_text)

    def extract_with_tables(self, pdf_path: str) -> Dict:
        """Extract content with tables (Docling only)."""
        if self._active_backend == "docling" and self._docling:
            return self._docling.extract_structured(pdf_path)
        
        # Fallback: just return sections
        return {
            "markdown": self.extract_full_text(pdf_path),
            "tables": [],
            "figures": [],
        }

    def parse_sections(self, text: str) -> Dict[str, dict]:
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
                sections[section_type.value] = {
                    "section_type": section_type.value,
                    "content": self._clean_section(content),
                    "order": list(SectionType).index(section_type)
                }

        sections["full_text"] = {
            "section_type": "full_text",
            "content": text,
            "order": 99
        }

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
    """Lightweight section extractor (backward compatible)."""

    def __init__(self, backend: str = "auto", gpu: bool = False):
        self.extractor = PDFExtractor(backend=backend, gpu=gpu)

    @property
    def available(self) -> bool:
        return self.extractor.available

    def extract_from_pdf(self, pdf_path: str) -> Dict[str, dict]:
        """Extract sections from PDF file."""
        return self.extractor.extract_sections(pdf_path)

    def extract_from_text(self, text: str) -> Dict[str, dict]:
        """Extract sections from text."""
        return self.extractor.parse_sections(text)

    def extract_abstract_from_metadata(self, abstract: str) -> Optional[dict]:
        """Create section dict from metadata abstract."""
        if not abstract:
            return None

        return {
            "section_type": "abstract",
            "content": abstract,
            "order": 0
        }