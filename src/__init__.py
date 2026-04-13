"""
VESTIGIUM - Zero Budget Aquatic Biomass Radar
Paquete principal
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__description__ = "WiFi-based biomass detection system using neural processing"

from .backend import signal_processor, neuromorphic_engine, slam_topological
from .utils import config, logger

__all__ = [
    "signal_processor",
    "neuromorphic_engine",
    "slam_topological",
    "config",
    "logger",
]
