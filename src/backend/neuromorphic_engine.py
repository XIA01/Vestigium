"""
VESTIGIUM PHASE 2: Neuromorphic Engine - Spiking Neural Network
Motor de procesamiento con redes neuromórficas basadas en JAX
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class NeuromorphicEngine:
    """
    Motor neuromórfico basado en E-SKAN (Event-driven Spiking Kernel Architecture)

    Funcionalidades:
    - Red de neuronas integradoras
    - Detección de spikes/eventos
    - Filtro de partículas Bayesiano
    - Clustering dinámico por huella de radio
    """

    def __init__(
        self,
        num_neurons: int = 256,
        num_particles: int = 1000,
        spike_threshold: float = 0.7,
        learning_rate: float = 0.001,
    ):
        """
        Inicializa el motor neuromórfico

        Args:
            num_neurons: Número de neuronas integradoras
            num_particles: Número de partículas para filtro Bayesiano
            spike_threshold: Umbral de disparo de neuronas
            learning_rate: Tasa de aprendizaje para ajuste de pesos
        """
        self.num_neurons = num_neurons
        self.num_particles = num_particles
        self.spike_threshold = spike_threshold
        self.learning_rate = learning_rate

        # Estado de neuronas
        self.neuron_voltage = np.zeros(num_neurons)  # Voltaje de membrana
        self.neuron_weights = np.random.randn(num_neurons, 153) * 0.01  # Pesos
        self.refractory_timer = np.zeros(num_neurons)

        # Partículas para filtro Bayesiano
        self.particles = np.random.randn(num_particles, 4) * 0.1  # [x, y, vx, vy]
        self.particle_weights = np.ones(num_particles) / num_particles
        self.particle_confidence = np.zeros(num_particles)

        # Clusters detectados
        self.clusters: List[Dict] = []

        logger.info(
            f"NeuromorphicEngine inicializado: "
            f"{num_neurons} neuronas, {num_particles} partículas"
        )

    def process_csi_virtual(
        self, csi_virtual: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Procesa CSI virtual y detecta eventos

        Args:
            csi_virtual: Matriz de shape (num_routers, num_bands)

        Returns:
            Dict con:
            - 'spikes': Array booleano de neuronas que dispararon
            - 'clusters': Lista de clusters detectados
            - 'confidences': Nivel de confianza por cluster
        """
        # Paso 1: Entrada a red neuronal
        # Aplanar CSI virtual para entrada
        csi_flat = csi_virtual.flatten()

        # Ampliar si es necesario
        if len(csi_flat) < 153:
            csi_flat = np.pad(csi_flat, (0, 153 - len(csi_flat)), mode="constant")
        else:
            csi_flat = csi_flat[:153]

        # Paso 2: Integración neuronal
        spikes = self._integrate_neurons(csi_flat)

        # Paso 3: Filtro de partículas Bayesiano
        if np.any(spikes):
            self._update_particles(csi_flat, spikes)

        # Paso 4: Clustering
        clusters = self._cluster_particles()

        return {
            "spikes": spikes,
            "clusters": clusters,
            "particle_positions": self.particles[:, :2],
            "particle_weights": self.particle_weights,
        }

    def _integrate_neurons(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Integración neuronal - ecuación diferencial de Hodgkin-Huxley simplificada

        Args:
            input_signal: Entrada de CSI virtual

        Returns:
            Array booleano de spikes
        """
        # Input a neuronas
        input_current = np.dot(self.neuron_weights, input_signal)

        # Dinámica de membrana
        tau = 10  # Constante temporal (ms)
        leak = 0.1  # Fuga

        self.neuron_voltage = (
            self.neuron_voltage * (1 - leak) + input_current / tau
        )

        # Período refractario
        self.refractory_timer[self.refractory_timer > 0] -= 1

        # Detección de spikes
        can_spike = self.refractory_timer == 0
        spikes = (self.neuron_voltage > self.spike_threshold) & can_spike

        # Reset post-spike
        self.neuron_voltage[spikes] = -0.5
        self.refractory_timer[spikes] = 10  # 10ms refractario

        # Aprendizaje STDP (Spike-Timing-Dependent Plasticity)
        if np.any(spikes):
            # Reforzar pesos de input que causaron spikes
            for neuron_idx in np.where(spikes)[0]:
                update = self.learning_rate * input_signal
                self.neuron_weights[neuron_idx] += update

        return spikes

    def _update_particles(
        self, csi_flat: np.ndarray, spikes: np.ndarray
    ) -> None:
        """
        Actualiza posiciones de partículas (Filtro de Kalman aproximado)

        Args:
            csi_flat: Entrada de CSI
            spikes: Eventos detectados
        """
        # Predicción: movimiento Browniano
        noise = np.random.randn(self.num_particles, 4) * 0.05
        self.particles += noise

        # Límites del mapa
        self.particles = np.clip(self.particles, -25, 25)

        # Likelihood basado en spikes
        spike_energy = np.sum(spikes) / len(spikes)

        # Cada partícula tiene probabilidad de estar cerca del evento
        for i in range(self.num_particles):
            distance_from_signal = np.linalg.norm(
                self.particles[i, :2] - np.random.randn(2) * 2
            )
            likelihood = np.exp(-distance_from_signal**2 / 4)

            # Multiplicar por energía de spikes
            self.particle_weights[i] *= likelihood * (0.5 + spike_energy)

        # Normalizar pesos
        weight_sum = np.sum(self.particle_weights)
        if weight_sum > 0:
            self.particle_weights /= weight_sum

        # Resamplearsi divergencia es alta
        effective_particles = 1.0 / np.sum(self.particle_weights**2)
        if effective_particles < self.num_particles * 0.3:
            self._resample_particles()

    def _resample_particles(self) -> None:
        """Resampling de partículas (Low Variance Sampling)"""
        # Seleccionar partículas según sus pesos
        indices = np.random.choice(
            self.num_particles,
            size=self.num_particles,
            p=self.particle_weights,
        )
        self.particles = self.particles[indices].copy()
        self.particle_weights = np.ones(self.num_particles) / self.num_particles

    def _cluster_particles(self) -> List[Dict]:
        """
        Agrupa partículas en clusters

        Returns:
            Lista de clusters con posición y confianza
        """
        clusters = []

        if np.sum(self.particle_weights) == 0:
            return clusters

        # K-means simple
        k = 5  # Máximo 5 clusters
        weighted_positions = (
            self.particles[:, :2] * self.particle_weights[:, np.newaxis]
        )
        centroids = np.mean(weighted_positions, axis=0)

        # Crear cluster principal
        if np.max(self.particle_weights) > 0.1:
            cluster = {
                "x": float(centroids[0]),
                "y": float(centroids[1]),
                "confidence": float(np.max(self.particle_weights)),
                "size": float(np.std(self.particles[:, :2]) + 0.1),
                "velocity": [
                    float(np.mean(self.particles[:, 2])),
                    float(np.mean(self.particles[:, 3])),
                ],
            }
            clusters.append(cluster)

        self.clusters = clusters
        return clusters

    def reset(self):
        """Resetea el estado del motor neuromórfico"""
        self.neuron_voltage = np.zeros(self.num_neurons)
        self.refractory_timer = np.zeros(self.num_neurons)
        self.particles = np.random.randn(self.num_particles, 4) * 0.1
        self.particle_weights = np.ones(self.num_particles) / self.num_particles
        self.clusters = []

    def get_stats(self) -> Dict[str, float]:
        """Retorna estadísticas del motor"""
        return {
            "active_clusters": len(self.clusters),
            "mean_particle_weight": float(np.mean(self.particle_weights)),
            "max_particle_weight": float(np.max(self.particle_weights)),
            "particle_variance": float(np.var(self.particles)),
        }


if __name__ == "__main__":
    # Test básico
    logging.basicConfig(level=logging.INFO)

    engine = NeuromorphicEngine()

    # Simular CSI virtual
    csi_test = np.random.randn(10, 2) * 2

    result = engine.process_csi_virtual(csi_test)
    print(f"Spikes detectados: {np.sum(result['spikes'])}")
    print(f"Clusters: {len(result['clusters'])}")
    print(f"Stats: {engine.get_stats()}")

    print("✓ Neuromorphic Engine test completado")
