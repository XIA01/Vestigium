"""
VESTIGIUM PHASE 3: Topological SLAM - Mapeo Emergente
Construye mapa dinámico de presencia sin lidar
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class SLAMTopological:
    """
    Sistema SLAM topológico emergente

    - Construye heatmap de probabilidad
    - Detecta obstáculos por exclusión
    - Auto-calibración estocástica
    """

    def __init__(
        self,
        map_width_m: float = 50,
        map_height_m: float = 50,
        resolution_cm: int = 10,
        decay_factor: float = 0.95,
    ):
        """
        Inicializa SLAM

        Args:
            map_width_m: Ancho del mapa en metros
            map_height_m: Alto del mapa en metros
            resolution_cm: Resolución de cada píxel en cm
            decay_factor: Factor de desvanecimiento (0-1)
        """
        self.map_width_m = map_width_m
        self.map_height_m = map_height_m
        self.resolution_cm = resolution_cm

        # Calcular tamaño de grid
        self.grid_width = int((map_width_m * 100) / resolution_cm)
        self.grid_height = int((map_height_m * 100) / resolution_cm)

        # Mapa de ocupancia log-odds
        self.occupancy_map = np.zeros((self.grid_height, self.grid_width))

        # Mapa de "tipo" (empty, solid, transit)
        self.obstacle_map = np.zeros((self.grid_height, self.grid_width))
        # 0 = empty, 1 = solid (pared), 2 = transit (pasillo)

        # Histórico de visitas
        self.visit_count = np.zeros((self.grid_height, self.grid_width))
        self.movement_count = np.zeros((self.grid_height, self.grid_width))

        # Parámetros
        self.decay_factor = decay_factor
        self.update_count = 0

        # Calibración
        self.baseline_signal_strength = -70  # dBm

        logger.info(
            f"SLAMTopological inicializado: "
            f"{self.grid_width}×{self.grid_height} grid, "
            f"{resolution_cm}cm resolution"
        )

    def update_from_clusters(
        self, clusters: List[Dict], delta_time_ms: float = 33
    ) -> Dict:
        """
        Actualiza mapa desde clusters detectados

        Args:
            clusters: Lista de clusters con {x, y, confidence, velocity}
            delta_time_ms: Tiempo desde última actualización

        Returns:
            Dict con estadísticas de actualización
        """
        # Desvanecimiento exponencial
        self.occupancy_map *= self.decay_factor
        self.movement_count *= self.decay_factor

        # Procesar cada cluster
        new_detections = 0
        for cluster in clusters:
            x = cluster.get("x", 0)
            y = cluster.get("y", 0)
            confidence = cluster.get("confidence", 0.5)
            velocity = cluster.get("velocity", [0, 0])

            # Convertir coordenadas a índices de grid
            grid_x, grid_y = self._world_to_grid(x, y)

            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                # Actualizar ocupancia con log-odds
                log_odds_update = np.log(confidence / (1 - confidence + 1e-6))
                self.occupancy_map[grid_y, grid_x] += log_odds_update

                # Registrar visita
                self.visit_count[grid_y, grid_x] += 1

                # Detectar movimiento
                speed = np.linalg.norm(velocity)
                if speed > 0.5:  # Threshold de movimiento
                    self.movement_count[grid_y, grid_x] += speed

                new_detections += 1

        # Clasificación de obstáculos
        self._classify_obstacles()

        # Incrementar contador
        self.update_count += 1

        return {
            "new_detections": new_detections,
            "total_clusters_processed": len(clusters),
            "active_cells": np.sum(self.occupancy_map > 0),
        }

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convierte coordenadas mundo a índices de grid

        Args:
            x, y: Coordenadas en metros (-25 a 25 típicamente)

        Returns:
            Tupla (grid_x, grid_y)
        """
        # Offset a centro de grid
        x_offset = x + self.map_width_m / 2
        y_offset = y + self.map_height_m / 2

        # Escalar a grid
        grid_x = int((x_offset * 100) / self.resolution_cm)
        grid_y = int((y_offset * 100) / self.resolution_cm)

        return grid_x, grid_y

    def _grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Inversa de world_to_grid"""
        x = (grid_x * self.resolution_cm / 100) - self.map_width_m / 2
        y = (grid_y * self.resolution_cm / 100) - self.map_height_m / 2
        return x, y

    def _classify_obstacles(self) -> None:
        """
        Clasifica cada píxel como:
        - 0: Empty (vacío)
        - 1: Solid (pared/mueble)
        - 2: Transit (pasillo/zona de movimiento)
        """
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                occupancy = 1.0 / (1.0 + np.exp(-self.occupancy_map[i, j]))

                if occupancy < 0.3:
                    self.obstacle_map[i, j] = 0  # Empty
                else:
                    # Distinguir entre solid y transit
                    visits = self.visit_count[i, j]
                    movement = self.movement_count[i, j]

                    if visits > 0:
                        movement_ratio = movement / visits

                        if movement_ratio < 0.3:
                            self.obstacle_map[i, j] = 1  # Solid
                        else:
                            self.obstacle_map[i, j] = 2  # Transit
                    else:
                        self.obstacle_map[i, j] = 0  # Unknown

    def get_occupancy_map(self) -> np.ndarray:
        """
        Retorna mapa de ocupancia como probabilidades

        Returns:
            Array (grid_height, grid_width) con valores [0, 1]
        """
        return 1.0 / (1.0 + np.exp(-self.occupancy_map))

    def get_heatmap(self) -> np.ndarray:
        """
        Retorna heatmap normalizado para visualización

        Returns:
            Array (grid_height, grid_width) con valores [0, 255]
        """
        occupancy = self.get_occupancy_map()
        heatmap = (occupancy * 255).astype(np.uint8)
        return heatmap

    def get_obstacle_map(self) -> np.ndarray:
        """Retorna mapa de tipos de obstáculos"""
        return self.obstacle_map.astype(np.uint8)

    def detect_static_regions(self) -> List[Dict]:
        """
        Detecta regiones sin movimiento (paredes/muebles)

        Returns:
            Lista de regiones sólidas con posición y tamaño
        """
        regions = []

        # Encontrar bloques conectados de tipo "solid"
        visited = np.zeros((self.grid_height, self.grid_width), dtype=bool)

        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if (
                    self.obstacle_map[i, j] == 1
                    and not visited[i, j]
                ):
                    # BFS para encontrar región conectada
                    region = self._flood_fill(i, j, visited, obstacle_type=1)
                    if region["size"] > 5:  # Mínimo tamaño
                        regions.append(region)

        return regions

    def _flood_fill(
        self, start_i: int, start_j: int, visited: np.ndarray, obstacle_type: int
    ) -> Dict:
        """
        Flood fill para encontrar regiones conectadas

        Returns:
            Dict con {center_x, center_y, size}
        """
        from collections import deque

        queue = deque([(start_i, start_j)])
        visited[start_i, start_j] = True
        cells = [(start_i, start_j)]

        while queue:
            i, j = queue.popleft()

            # Vecinos
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if (
                    0 <= ni < self.grid_height
                    and 0 <= nj < self.grid_width
                    and not visited[ni, nj]
                    and self.obstacle_map[ni, nj] == obstacle_type
                ):
                    visited[ni, nj] = True
                    queue.append((ni, nj))
                    cells.append((ni, nj))

        # Calcular centroide
        cells_array = np.array(cells)
        center_grid_i = np.mean(cells_array[:, 0])
        center_grid_j = np.mean(cells_array[:, 1])

        center_x, center_y = self._grid_to_world(
            int(center_grid_j), int(center_grid_i)
        )

        return {
            "center_x": float(center_x),
            "center_y": float(center_y),
            "size": len(cells),
        }

    def reset(self):
        """Reinicia el mapa"""
        self.occupancy_map.fill(0)
        self.obstacle_map.fill(0)
        self.visit_count.fill(0)
        self.movement_count.fill(0)
        self.update_count = 0

    def get_stats(self) -> Dict:
        """Estadísticas del SLAM"""
        occupancy = self.get_occupancy_map()
        return {
            "occupied_cells": int(np.sum(occupancy > 0.5)),
            "obstacle_cells": int(np.sum(self.obstacle_map > 0)),
            "mean_occupancy": float(np.mean(occupancy)),
            "updates": self.update_count,
        }


if __name__ == "__main__":
    # Test básico
    logging.basicConfig(level=logging.INFO)

    slam = SLAMTopological()

    # Simular clusters
    for i in range(50):
        cluster = {
            "x": np.sin(i * 0.1) * 5,
            "y": np.cos(i * 0.1) * 5,
            "confidence": 0.7,
            "velocity": [np.cos(i * 0.1) * 0.5, np.sin(i * 0.1) * 0.5],
        }

        slam.update_from_clusters([cluster])

    print(f"Mapa ocupancia shape: {slam.get_occupancy_map().shape}")
    print(f"Stats: {slam.get_stats()}")
    print("✓ SLAM Topological test completado")
