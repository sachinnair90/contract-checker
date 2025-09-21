"""
Contract scanning utilities using docling for PDF processing.
"""

import os
from pathlib import Path
from typing import Dict, Any

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


class ContractScanner:
    """Utility class for scanning contract documents using docling."""

    def __init__(self):
        """Initialize the contract scanner with optimized settings."""
        # Set up accelerator options
        accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=(
                AcceleratorDevice.CUDA
                if self._cuda_available()
                else AcceleratorDevice.CPU
            ),
        )

        # Configure pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True

        # Create converter
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

    def _cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def scan_contract(self, pdf_path: str) -> Dict[str, Any]:
        """
        Scan a contract PDF and extract content information.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing scan results
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Convert the document
        conversion_result = self.converter.convert(pdf_path)

        # Extract information
        document = conversion_result.document
        markdown_content = document.export_to_markdown()

        result = {
            "filename": Path(pdf_path).name,
            "pages": len(document.pages),
            "tables": len(document.tables),
            "content": markdown_content,
            "has_content": len(markdown_content.strip()) > 0,
        }

        return result
