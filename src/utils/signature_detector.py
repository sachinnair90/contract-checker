"""
Signature detection utilities using YOLO and DETR models.
"""

import os
from pathlib import Path
from typing import Dict, Any
import numpy as np
import supervision as sv
import pymupdf as fitz
import torch

from ultralytics import YOLO
from huggingface_hub import hf_hub_download, login
from transformers import AutoImageProcessor, AutoModelForObjectDetection


class SignatureDetector:
    """Utility class for detecting signatures in contract documents."""

    def __init__(self, model_type: str = "yolo", hf_token: str = None):
        """
        Initialize the signature detector.

        Args:
            model_type: Type of model to use ('yolo' or 'detr')
            hf_token: HuggingFace token for downloading models
        """
        self.model_type = model_type.lower()
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable is required")

        # Initialize model attributes
        self.yolo_model = None
        self.detr_processor = None
        self.detr_model = None

        # Validate model type
        if self.model_type not in ["yolo", "detr"]:
            raise ValueError("model_type must be 'yolo' or 'detr'")

    def _load_yolo_model(self):
        """Load the YOLO model if not already loaded."""
        if self.yolo_model is None:
            login(self.hf_token)
            model_path = hf_hub_download(
                repo_id="tech4humans/yolov8s-signature-detector", filename="yolov8s.pt"
            )
            self.yolo_model = YOLO(model_path)

    def _load_detr_model(self):
        """Load the DETR model if not already loaded."""
        if self.detr_processor is None or self.detr_model is None:
            self.detr_processor = AutoImageProcessor.from_pretrained(
                "tech4humans/conditional-detr-50-signature-detector", use_fast=True
            )
            self.detr_model = AutoModelForObjectDetection.from_pretrained(
                "tech4humans/conditional-detr-50-signature-detector"
            )

    def detect_signatures(self, pdf_path: str) -> Dict[str, Any]:
        """
        Detect signatures in a PDF document.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing detection results
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if self.model_type == "yolo":
            return self._detect_with_yolo(pdf_path)
        elif self.model_type == "detr":
            return self._detect_with_detr(pdf_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _detect_with_yolo(self, pdf_path: str) -> Dict[str, Any]:
        """Detect signatures using YOLO model."""
        self._load_yolo_model()

        detected_signatures = []
        doc = fitz.open(pdf_path)

        for page_num in range(doc.page_count):
            page = doc[page_num]
            pix = page.get_pixmap()
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            results = self.yolo_model(img)
            detections = sv.Detections.from_ultralytics(results[0])

            # Filter detections with confidence > 0.75
            detections = detections[detections.confidence > 0.75]

            if len(detections) > 0:
                for xyxy in detections.xyxy:
                    # Convert coordinates to integers
                    x1, y1, x2, y2 = map(int, xyxy)

                    # Ensure coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img.shape[1], x2)
                    y2 = min(img.shape[0], y2)

                    signature_info = {
                        "page": page_num + 1,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(
                            detections.confidence[list(detections.xyxy).index(xyxy)]
                        ),
                        "model": "yolo",
                    }
                    detected_signatures.append(signature_info)

        doc.close()

        result = {
            "filename": Path(pdf_path).name,
            "model": "yolo",
            "total_signatures": len(detected_signatures),
            "signatures": detected_signatures,
            "pages_with_signatures": list(
                set(sig["page"] for sig in detected_signatures)
            ),
            "has_signatures": len(detected_signatures) > 0,
        }

        return result

    def _detect_with_detr(self, pdf_path: str) -> Dict[str, Any]:
        """Detect signatures using DETR model."""
        self._load_detr_model()

        detected_signatures = []
        doc = fitz.open(pdf_path)

        for page_num in range(doc.page_count):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Increase resolution
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            inputs = self.detr_processor(images=img, return_tensors="pt")

            with torch.no_grad():
                outputs = self.detr_model(**inputs)

            target_sizes = torch.tensor([img.shape[:2]])
            detections = self.detr_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.75
            )[0]

            # Filter detections
            scores = detections["scores"]
            labels = detections["labels"]
            boxes = detections["boxes"]

            filtered_indices = scores > 0.75

            for score, label, box in zip(
                scores[filtered_indices],
                labels[filtered_indices],
                boxes[filtered_indices],
            ):
                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, box.tolist())

                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img.shape[1], x2)
                y2 = min(img.shape[0], y2)

                signature_info = {
                    "page": page_num + 1,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(score.item()),
                    "model": "detr",
                }
                detected_signatures.append(signature_info)

        doc.close()

        result = {
            "filename": Path(pdf_path).name,
            "model": "detr",
            "total_signatures": len(detected_signatures),
            "signatures": detected_signatures,
            "pages_with_signatures": list(
                set(sig["page"] for sig in detected_signatures)
            ),
            "has_signatures": len(detected_signatures) > 0,
        }

        return result
