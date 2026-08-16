"""Deployment backends and data structures for the public MeowID API."""

from .registry import EmbeddingRegistry
from .types import EmbeddingResult, PoseDetection, PredictionResult, SearchMatch

__all__ = [
    "EmbeddingRegistry",
    "EmbeddingResult",
    "PoseDetection",
    "PredictionResult",
    "SearchMatch",
]
