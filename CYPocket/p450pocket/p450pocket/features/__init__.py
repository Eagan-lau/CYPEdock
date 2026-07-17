"""PocketMiner feature extraction public API.

TensorFlow is imported lazily when ``PocketMinerExtractor`` is instantiated,
so structure/label preparation remains usable in a PyTorch-only environment.
"""

from .extractor import (
    FeatureExtractionError,
    FeatureResult,
    PocketMinerConfig,
    PocketMinerExtractor,
)

__all__ = [
    "FeatureExtractionError",
    "FeatureResult",
    "PocketMinerConfig",
    "PocketMinerExtractor",
]
