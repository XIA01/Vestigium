"""
VESTIGIUM Main - Punto de entrada del sistema
Orquesta las 4 fases del procesamiento
"""

import numpy as np
import time
from typing import Dict, Optional
import logging

from .backend import SignalProcessor, NeuromorphicEngine, SLAMTopological
from .utils import get_config, get_logger, setup_logging

logger = get_logger("vestigium.main")


class VestigiumSystem:
    """
    Sistema completo VESTIGIUM
    Orquesta Signal Processing → Neuromorphic → SLAM → Visualization
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa el sistema VESTIGIUM

        Args:
            config_path: Ruta al archivo config.yaml
        """
        self.config = get_config(config_path)
        logger.info("VESTIGIUM System inicializado")

        # Fase 1: Signal Processor
        self.signal_processor = SignalProcessor(
            num_routers=153,
            num_bands=2,
            window_size_ms=self.config.get(
                "signal_processing.scintillation.window_size_ms", 100
            ),
            sampling_rate_hz=self.config.get(
                "hardware.polling.max_samples_per_second", 100
            ),
        )

        # Fase 2: Neuromorphic Engine
        self.neuromorphic_engine = NeuromorphicEngine(
            num_neurons=self.config.get("neuromorphic.skan_network.num_neurons", 256),
            num_particles=self.config.get("neuromorphic.particle_filter.num_particles", 1000),
            spike_threshold=self.config.get("neuromorphic.skan_network.spike_threshold", 0.7),
            learning_rate=self.config.get(
                "neuromorphic.skan_network.learning_rate", 0.001
            ),
        )

        # Fase 3: SLAM Topológico
        self.slam = SLAMTopological(
            map_width_m=self.config.get("slam.map.width_meters", 50),
            map_height_m=self.config.get("slam.map.height_meters", 50),
            resolution_cm=self.config.get("slam.map.resolution_cm", 10),
            decay_factor=self.config.get("slam.heatmap.decay_factor", 0.95),
        )

        # Estadísticas
        self.frame_count = 0
        self.total_detections = 0
        self.fps_clock = time.time()

    def process_frame(self, rssi_data: np.ndarray) -> Dict:
        """
        Procesa un frame RSSI a través de todas las fases

        Args:
            rssi_data: Array de RSSI shape (num_routers, num_bands)

        Returns:
            Dict con resultados de todas las fases
        """
        self.frame_count += 1

        # Fase 1: Signal Processing
        signal_result = self.signal_processor.process_rssi(rssi_data)

        if signal_result["csi_virtual"] is None:
            # Aún no hay suficientes datos
            return {
                "frame": self.frame_count,
                "status": "buffering",
                "data": None,
            }

        # Fase 2: Neuromorphic Engine
        neuro_result = self.neuromorphic_engine.process_csi_virtual(
            signal_result["csi_virtual"]
        )

        # Fase 3: SLAM
        slam_result = self.slam.update_from_clusters(neuro_result["clusters"])

        # Agregar detecciones
        if neuro_result["clusters"]:
            self.total_detections += len(neuro_result["clusters"])

        # Compilar resultado final
        output = {
            "frame": self.frame_count,
            "status": "ok",
            "timestamp": time.time(),
            "phase1": {
                "csi_shape": signal_result["csi_virtual"].shape,
                "band_ratio": signal_result["band_ratio"],
            },
            "phase2": {
                "spikes": int(np.sum(neuro_result["spikes"])),
                "clusters": neuro_result["clusters"],
                "particle_positions": neuro_result["particle_positions"],
            },
            "phase3": {
                "heatmap": self.slam.get_heatmap(),
                "obstacle_map": self.slam.get_obstacle_map(),
                "occupancy": slam_result,
            },
        }

        return output

    def get_visualization_data(self) -> Dict:
        """
        Obtiene datos para visualización

        Returns:
            Dict con heatmap, clusters, etc
        """
        return {
            "heatmap": self.slam.get_heatmap(),
            "obstacle_map": self.slam.get_obstacle_map(),
            "clusters": self.neuromorphic_engine.clusters,
            "particles": self.neuromorphic_engine.particles[:, :2],
        }

    def get_stats(self) -> Dict:
        """Retorna estadísticas del sistema"""
        elapsed = time.time() - self.fps_clock
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        return {
            "frames_processed": self.frame_count,
            "total_detections": self.total_detections,
            "fps": fps,
            "signal_processor": self.signal_processor.get_stats(),
            "neuromorphic": self.neuromorphic_engine.get_stats(),
            "slam": self.slam.get_stats(),
        }

    def reset(self):
        """Reinicia todos los módulos"""
        self.signal_processor.reset_baseline()
        self.neuromorphic_engine.reset()
        self.slam.reset()
        self.frame_count = 0
        self.total_detections = 0
        logger.info("Sistema reiniciado")


def main():
    """Función principal para testing"""
    # Setup logging
    setup_logging(level="INFO", log_file="logs/vestigium.log")

    logger.info("=" * 60)
    logger.info("VESTIGIUM - Zero Budget Aquatic Biomass Radar")
    logger.info("=" * 60)

    try:
        # Inicializar sistema
        system = VestigiumSystem()

        # Simular stream de RSSI
        logger.info("Iniciando simulación...")
        for frame_idx in range(500):
            # Simular RSSI con patrón gaussiano
            rssi_data = np.random.normal(-60, 5, size=(153, 2))

            # Agregar patrón sintetizado
            if 100 < frame_idx < 400:
                # Objeto moviéndose en círculo
                angle = (frame_idx - 100) * 0.02
                x_perturb = np.sin(angle) * 3
                y_perturb = np.cos(angle) * 3
                rssi_data[:10, :] -= (x_perturb + y_perturb) * 2

            # Procesar frame
            result = system.process_frame(rssi_data)

            if frame_idx % 100 == 0:
                stats = system.get_stats()
                logger.info(f"Frame {frame_idx}: FPS={stats['fps']:.2f}, "
                           f"Detecciones={stats['total_detections']}")

        # Estadísticas finales
        logger.info("\n" + "=" * 60)
        logger.info("ESTADÍSTICAS FINALES")
        logger.info("=" * 60)
        final_stats = system.get_stats()
        for key, value in final_stats.items():
            logger.info(f"{key}: {value}")

        logger.info("✓ Simulación completada exitosamente")

    except Exception as e:
        logger.error(f"Error en simulación: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
