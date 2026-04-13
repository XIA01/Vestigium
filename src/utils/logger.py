"""
VESTIGIUM Logging System
Sistema centralizado de logging con soporte para múltiples niveles
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    from loguru import logger as loguru_logger
    HAS_LOGURU = True
except ImportError:
    HAS_LOGURU = False


class VestigiumLogger:
    """Logger centralizado para VESTIGIUM"""

    _loggers = {}

    @staticmethod
    def get_logger(name: str, level: str = "INFO") -> logging.Logger:
        """
        Obtiene o crea un logger para un módulo

        Args:
            name: Nombre del módulo (ej: "signal_processor")
            level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        Returns:
            Logger configurado
        """
        if name in VestigiumLogger._loggers:
            return VestigiumLogger._loggers[name]

        logger = logging.getLogger(f"vestigium.{name}")
        logger.setLevel(level)

        # Handler a console
        if not logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)

            # Formato colorido
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        VestigiumLogger._loggers[name] = logger
        return logger

    @staticmethod
    def setup_file_logging(
        log_file: str = "logs/vestigium.log", level: str = "INFO"
    ):
        """
        Configura logging a archivo

        Args:
            log_file: Ruta del archivo de log
            level: Nivel de logging
        """
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Handler a archivo para root logger
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)

        # Agregar a todos los loggers de VESTIGIUM
        for logger in VestigiumLogger._loggers.values():
            logger.addHandler(file_handler)

        logging.getLogger("vestigium").addHandler(file_handler)


# Logger por defecto
def get_logger(name: str = "main") -> logging.Logger:
    """Crea/obtiene logger con nombre específico"""
    return VestigiumLogger.get_logger(name)


# Setup rápido para scripts
def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    verbose: bool = False,
):
    """
    Setup rápido de logging

    Args:
        level: Nivel de logging
        log_file: Archivo de salida (opcional)
        verbose: Si True, muestra más detalles
    """
    if verbose:
        level = "DEBUG"

    root_logger = logging.getLogger("vestigium")
    root_logger.setLevel(level)

    if log_file:
        VestigiumLogger.setup_file_logging(log_file, level)


# Ejemplo de uso:
if __name__ == "__main__":
    # Setup
    setup_logging(level="DEBUG", log_file="test.log")

    # Usar logger
    logger = get_logger("test_module")
    logger.info("Mensaje informativo")
    logger.debug("Mensaje de debug")
    logger.warning("Mensaje de advertencia")
    logger.error("Mensaje de error")

    print("✓ Logger configurado correctamente")
