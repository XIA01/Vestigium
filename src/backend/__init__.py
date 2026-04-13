"""
VESTIGIUM Backend - Núcleo de procesamiento
Contiene los módulos de las 4 fases
"""

from .signal_processor import SignalProcessor
from .neuromorphic_engine import NeuromorphicEngine
from .slam_topological import SLAMTopological

__all__ = ["SignalProcessor", "NeuromorphicEngine", "SLAMTopological"]
