"""
VESTIGIUM PHASE 1: Signal Processor - CSI Virtual Extraction
Extrae "CSI Virtual" del análisis de varianza de RSSI
"""

import numpy as np
from typing import Tuple, Dict, Optional
from collections import deque
from scipy import signal as scipy_signal
import logging

logger = logging.getLogger(__name__)


class SignalProcessor:
    """
    Procesa señales WiFi para extraer CSI virtual

    Pasos:
    1. Buffer circular de RSSI
    2. Análisis de varianza temporal
    3. FFT para detectar patrones
    4. Fusión de bandas (2.4GHz + 5GHz)
    """

    def __init__(
        self,
        num_routers: int = 153,
        num_bands: int = 2,
        window_size_ms: int = 100,
        sampling_rate_hz: int = 100,
    ):
        """
        Inicializa el procesador de señales

        Args:
            num_routers: Número de routers WiFi a monitorear
            num_bands: Número de bandas (2.4GHz, 5GHz)
            window_size_ms: Tamaño de ventana para análisis
            sampling_rate_hz: Frecuencia de muestreo RSSI
        """
        self.num_routers = num_routers
        self.num_bands = num_bands
        self.window_size_ms = window_size_ms
        self.sampling_rate_hz = sampling_rate_hz

        # Calcular tamaño de buffer
        self.window_samples = int(window_size_ms * sampling_rate_hz / 1000)

        # Buffer circular para RSSI
        self.rssi_buffer = deque(
            maxlen=self.window_samples * 5  # 5 ventanas de histórico
        )

        # Baseline de varianza (se calibra automáticamente)
        self.baseline_variance = None
        self.baseline_samples = 0

        logger.info(
            f"SignalProcessor inicializado: "
            f"{num_routers} routers × {num_bands} bandas, "
            f"ventana={window_size_ms}ms, fs={sampling_rate_hz}Hz"
        )

    def process_rssi(
        self, rssi_matrix: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Procesa matriz de RSSI y extrae CSI virtual

        Args:
            rssi_matrix: Array de shape (num_routers, num_bands)
                        Valores RSSI en dBm

        Returns:
            Dict con:
            - 'csi_virtual': Matriz de varianza CSI
            - 'power_spectrum': Espectro de potencia FFT
            - 'band_ratio': Diferencia 2.4GHz vs 5GHz
        """
        # Agregar nueva muestra al buffer
        self.rssi_buffer.append(rssi_matrix.copy())

        if len(self.rssi_buffer) < self.window_samples:
            # Buffer aún no lleno
            return {
                "csi_virtual": None,
                "power_spectrum": None,
                "band_ratio": None,
            }

        # Convertir buffer a array
        rssi_history = np.array(list(self.rssi_buffer))
        # Shape: (buffer_len, num_routers, num_bands)

        # Fase 1.1: Análisis de varianza
        csi_virtual = self._compute_variance(rssi_history)

        # Fase 1.2: Análisis espectral FFT
        power_spectrum = self._compute_power_spectrum(rssi_history)

        # Fase 1.3: Fusión de bandas
        band_ratio = self._compute_band_ratio(rssi_history)

        return {
            "csi_virtual": csi_virtual,
            "power_spectrum": power_spectrum,
            "band_ratio": band_ratio,
            "timestamp": rssi_matrix,
        }

    def _compute_variance(self, rssi_history: np.ndarray) -> np.ndarray:
        """
        Calcula varianza temporal de RSSI

        Args:
            rssi_history: Array de shape (time, routers, bands)

        Returns:
            Varianza normalizada, shape (routers, bands)
        """
        # Ventanas móviles
        variance = np.zeros((self.num_routers, self.num_bands))

        for router_idx in range(self.num_routers):
            for band_idx in range(self.num_bands):
                # Extraer serie temporal para este router/banda
                series = rssi_history[:, router_idx, band_idx]

                # Calcular varianza
                local_variance = np.var(series)

                # Normalizar contra baseline
                if self.baseline_variance is not None:
                    variance[router_idx, band_idx] = (
                        local_variance - self.baseline_variance[router_idx, band_idx]
                    ) / (self.baseline_variance[router_idx, band_idx] + 1e-6)
                else:
                    variance[router_idx, band_idx] = local_variance

        # Auto-calibración: actualizar baseline
        self.baseline_samples += 1
        if self.baseline_samples < 100:  # Primeras 100 muestras
            if self.baseline_variance is None:
                self.baseline_variance = np.var(rssi_history, axis=0)
            else:
                # Promedio móvil del baseline
                alpha = 1.0 / self.baseline_samples
                new_baseline = np.var(rssi_history, axis=0)
                self.baseline_variance = (
                    1 - alpha
                ) * self.baseline_variance + alpha * new_baseline

        return variance

    def _compute_power_spectrum(
        self, rssi_history: np.ndarray
    ) -> np.ndarray:
        """
        Calcula espectro de potencia via FFT

        Args:
            rssi_history: Array de shape (time, routers, bands)

        Returns:
            Power spectrum, shape (routers, bands, freq_bins)
        """
        power_spectrum = np.zeros(
            (self.num_routers, self.num_bands, self.window_samples // 2)
        )

        for router_idx in range(self.num_routers):
            for band_idx in range(self.num_bands):
                series = rssi_history[:, router_idx, band_idx]

                # Aplicar ventana Hann para reducir leakage
                windowed = series * np.hanning(len(series))

                # FFT
                fft_result = np.fft.fft(windowed)
                magnitude = np.abs(fft_result[: len(fft_result) // 2])

                # Power spectrum (magnitud al cuadrado)
                power_spectrum[router_idx, band_idx, :] = magnitude**2

        return power_spectrum

    def _compute_band_ratio(self, rssi_history: np.ndarray) -> np.ndarray:
        """
        Calcula razón de atenuación entre bandas

        Indicador de profundidad/tamaño de objeto

        Args:
            rssi_history: Array de shape (time, routers, bands)

        Returns:
            Ratio 2.4GHz/5GHz por router
        """
        band_2_4ghz = rssi_history[:, :, 0]  # banda 2.4
        band_5ghz = rssi_history[:, :, 1]  # banda 5

        # Promedios en tiempo
        mean_2_4 = np.mean(band_2_4ghz, axis=0)
        mean_5 = np.mean(band_5ghz, axis=0)

        # Ratio (evitar división por cero)
        ratio = mean_2_4 / (mean_5 + 1e-6)

        return ratio

    def reset_baseline(self):
        """Reinicia calibración de baseline"""
        self.baseline_variance = None
        self.baseline_samples = 0
        logger.info("Baseline reseteado")

    def get_stats(self) -> Dict[str, float]:
        """Retorna estadísticas del procesador"""
        return {
            "buffer_size": len(self.rssi_buffer),
            "baseline_samples": self.baseline_samples,
            "window_samples": self.window_samples,
        }


if __name__ == "__main__":
    # Test básico
    logging.basicConfig(level=logging.INFO)

    processor = SignalProcessor(num_routers=10, num_bands=2)

    # Simular RSSI aleatorio
    for i in range(200):
        # RSSI típicamente entre -30 y -90 dBm
        rssi_data = np.random.normal(-60, 5, size=(10, 2))
        result = processor.process_rssi(rssi_data)

        if result["csi_virtual"] is not None:
            print(f"Iteración {i}: CSI Virtual shape={result['csi_virtual'].shape}")
            break

    print("✓ Signal Processor test completado")
