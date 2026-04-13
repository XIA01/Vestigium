"""
VESTIGIUM Configuration Loader
Carga y valida la configuración desde YAML
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseSettings, validator
import logging

logger = logging.getLogger(__name__)


class VestigiumConfig:
    """Configuración centralizada de VESTIGIUM"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa configuración desde archivo YAML

        Args:
            config_path: Ruta al archivo config.yaml
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self):
        """Carga y valida configuración desde YAML"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Archivo de configuración no encontrado: {self.config_path}\n"
                f"Crea el archivo copiando config.example.yaml"
            )

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Configuración cargada desde {self.config_path}")
            self._validate_config()
        except yaml.YAMLError as e:
            raise ValueError(f"Error al parsear YAML: {e}")

    def _validate_config(self):
        """Valida que la configuración tenga los campos requeridos"""
        required_sections = [
            "hardware",
            "signal_processing",
            "neuromorphic",
            "slam",
            "visualization",
            "logging",
        ]

        for section in required_sections:
            if section not in self.config:
                logger.warning(
                    f"Sección '{section}' no encontrada en configuración. "
                    f"Usando valores por defecto."
                )

    def get(self, path: str, default: Any = None) -> Any:
        """
        Obtiene valor de configuración usando notación de puntos

        Args:
            path: Ruta tipo 'hardware.router.ip'
            default: Valor por defecto si no existe

        Returns:
            Valor de configuración
        """
        keys = path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def __getitem__(self, key: str) -> Any:
        """Acceso tipo diccionario"""
        return self.get(key)

    def __repr__(self) -> str:
        return f"VestigiumConfig(path={self.config_path})"


# Instancia global de configuración
_config_instance = None


def get_config(config_path: str = "config.yaml") -> VestigiumConfig:
    """
    Obtiene instancia singleton de configuración

    Args:
        config_path: Ruta al archivo config.yaml (solo usado en primer llamado)

    Returns:
        Instancia de VestigiumConfig
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = VestigiumConfig(config_path)
    return _config_instance


def reload_config(config_path: str = "config.yaml"):
    """Recarga la configuración"""
    global _config_instance
    _config_instance = VestigiumConfig(config_path)


# Configuraciones preconfiguradas
DEFAULT_CONFIG = {
    "hardware": {
        "router": {
            "ip": "192.168.1.1",
            "port": 80,
            "username": "admin",
            "password": "admin",
        },
        "gpu": {"enabled": True, "device": "cuda:0", "memory_fraction": 0.9},
        "polling": {"max_samples_per_second": 100, "timeout_seconds": 5},
    },
    "signal_processing": {
        "band_2_4ghz": {
            "enabled": True,
            "channels": [1, 6, 11],
            "smoothing_window": 10,
        },
        "band_5ghz": {
            "enabled": True,
            "channels": [36, 40, 44, 48, 149, 153, 157, 161],
            "smoothing_window": 10,
        },
        "scintillation": {
            "high_frequency_cutoff": 20,
            "window_size_ms": 100,
            "sensitivity": 0.5,
        },
    },
    "neuromorphic": {
        "backend": "jax",
        "precision": "float32",
        "particle_filter": {"num_particles": 1000, "resampling_threshold": 0.5},
        "skan_network": {
            "num_neurons": 256,
            "spike_threshold": 0.7,
            "refractory_period_ms": 10,
            "learning_rate": 0.001,
        },
        "clustering": {
            "algorithm": "kmeans",
            "max_clusters": 10,
            "min_cluster_size": 2,
        },
    },
    "slam": {
        "map": {
            "width_meters": 50,
            "height_meters": 50,
            "resolution_cm": 10,
            "dynamic": True,
        },
        "heatmap": {"decay_factor": 0.95, "peak_temperature": 1.0},
    },
    "visualization": {
        "server": {"host": "0.0.0.0", "port": 5000, "debug": False},
        "canvas": {"width": 1200, "height": 800, "fps": 30, "motion_blur": True},
    },
    "logging": {"level": "INFO", "file": "logs/vestigium.log"},
}


if __name__ == "__main__":
    # Test de carga de configuración
    try:
        config = get_config()
        print(f"✓ Configuración cargada exitosamente")
        print(f"  Router IP: {config.get('hardware.router.ip')}")
        print(f"  GPU Enabled: {config.get('hardware.gpu.enabled')}")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
