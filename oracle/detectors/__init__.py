"""The two consistency detectors of Figure 7."""
from .functional import FunctionalConsistencyDetector
from .interaction import StateInteractionConsistencyDetector

__all__ = ["FunctionalConsistencyDetector", "StateInteractionConsistencyDetector"]
