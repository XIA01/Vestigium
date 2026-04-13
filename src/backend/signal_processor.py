"""
VESTIGIUM PHASE 1: Signal Processor - CSI Virtual Extraction
Full JAX implementation for GPU acceleration.
Zero Python loops. All operations vectorized on GPU.
"""

import jax
import jax.numpy as jnp
from jax import jit, lax
from typing import Dict, NamedTuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SignalProcessorState(NamedTuple):
    """State container for signal processor."""
    rssi_buffer: jnp.ndarray  # (buffer_size, num_routers, num_bands)
    baseline_variance: jnp.ndarray  # (num_routers, num_bands)
    baseline_samples: int
    buffer_index: int


class SignalProcessor:
    """
    GPU-accelerated signal processing for WiFi RSSI.

    All computation is JAX-jitted for maximum performance.
    Maintains ring buffer on GPU, computes variance/FFT/band-ratio in one pass.
    """

    def __init__(
        self,
        num_routers: int = 153,
        num_bands: int = 2,
        window_size_ms: int = 100,
        sampling_rate_hz: int = 100,
    ):
        """
        Initialize signal processor.

        Args:
            num_routers: Number of WiFi APs
            num_bands: 2 = [2.4GHz, 5GHz]
            window_size_ms: Analysis window size
            sampling_rate_hz: RSSI sampling rate
        """
        self.num_routers = num_routers
        self.num_bands = num_bands
        self.window_size_ms = window_size_ms
        self.sampling_rate_hz = sampling_rate_hz

        # Buffer size: keep 5 windows of history
        self.window_samples = int(window_size_ms * sampling_rate_hz / 1000)
        self.buffer_size = self.window_samples * 5

        # Initialize state on GPU
        self.state = SignalProcessorState(
            rssi_buffer=jnp.zeros((self.buffer_size, num_routers, num_bands), dtype=jnp.float32),
            baseline_variance=jnp.ones((num_routers, num_bands), dtype=jnp.float32) * 2.0,
            baseline_samples=0,
            buffer_index=0,
        )

        # Compile JAX functions
        self._compiled_process = jit(self._process_jit)
        self._compiled_update_buffer = jit(self._update_buffer_jit)
        self._compiled_update_baseline = jit(self._update_baseline_jit)

        logger.info(
            f"SignalProcessor (JAX GPU): {num_routers} routers × {num_bands} bands, "
            f"buffer={self.buffer_size} samples, window={self.window_samples}"
        )

    def process_rssi(self, rssi_data: np.ndarray) -> Dict:
        """
        Process new RSSI sample.

        Args:
            rssi_data: np.ndarray shape (num_routers, num_bands) in dBm

        Returns:
            Dict with CSI virtual, power spectrum, band ratio
        """
        # Convert to JAX array
        rssi_jax = jnp.asarray(rssi_data, dtype=jnp.float32)

        # Update buffer (ring buffer operation on GPU)
        self.state = self._compiled_update_buffer(self.state, rssi_jax)

        # Update baseline (exponential moving average)
        self.state = self._compiled_update_baseline(self.state, rssi_jax)

        # If buffer not full yet, return None
        if self.state.baseline_samples < self.window_samples:
            return {
                "csi_virtual": None,
                "power_spectrum": None,
                "band_ratio": None,
            }

        # Process full pipeline
        csi_virtual, power_spectrum, band_ratio = self._compiled_process(self.state)

        return {
            "csi_virtual": np.asarray(csi_virtual),  # Return as numpy
            "power_spectrum": np.asarray(power_spectrum),
            "band_ratio": np.asarray(band_ratio),
            "timestamp": rssi_data,
        }

    @staticmethod
    def _update_buffer_jit(state: SignalProcessorState, rssi_new: jnp.ndarray) -> SignalProcessorState:
        """Ring buffer update (JAX JIT)."""
        idx = state.buffer_index % state.rssi_buffer.shape[0]
        rssi_buffer_new = state.rssi_buffer.at[idx].set(rssi_new)
        return state._replace(
            rssi_buffer=rssi_buffer_new,
            buffer_index=state.buffer_index + 1,
        )

    @staticmethod
    def _update_baseline_jit(state: SignalProcessorState, rssi_new: jnp.ndarray) -> SignalProcessorState:
        """Update baseline variance (JAX JIT)."""
        def do_calibration(_):
            # First 100 samples: compute mean variance
            new_var = jnp.var(rssi_new)  # Scalar
            alpha = 1.0 / (state.baseline_samples + 1)
            baseline_new = (1 - alpha) * state.baseline_variance + alpha * new_var
            return baseline_new, state.baseline_samples + 1

        def no_calibration(_):
            return state.baseline_variance, state.baseline_samples

        # Conditional: only update first 100 samples
        baseline_new, samples_new = lax.cond(
            state.baseline_samples < 100,
            do_calibration,
            no_calibration,
            None,
        )

        return state._replace(
            baseline_variance=baseline_new,
            baseline_samples=samples_new,
        )

    @staticmethod
    def _process_jit(state: SignalProcessorState):
        """
        Main processing pipeline (JAX JIT).
        Variance, FFT, band ratio—all on GPU.
        """
        buffer = state.rssi_buffer

        # Variance: (buffer_size, num_routers, num_bands) → (num_routers, num_bands)
        variance = jnp.var(buffer, axis=0)
        variance_normalized = (variance - state.baseline_variance) / (state.baseline_variance + 1e-6)

        # Power spectrum: FFT along time axis
        fft_result = jnp.fft.rfft(buffer, axis=0)
        power_spectrum = jnp.abs(fft_result) ** 2

        # Band ratio: mean 2.4GHz / mean 5GHz
        mean_2_4 = jnp.mean(buffer[:, :, 0], axis=0)  # (num_routers,)
        mean_5 = jnp.mean(buffer[:, :, 1], axis=0)
        band_ratio = mean_2_4 / (mean_5 + 1e-6)  # (num_routers,)

        return variance_normalized, power_spectrum, band_ratio

    def get_stats(self) -> Dict:
        """Runtime statistics."""
        return {
            "buffer_index": int(self.state.buffer_index),
            "baseline_samples": int(self.state.baseline_samples),
            "buffer_size": self.buffer_size,
            "window_samples": self.window_samples,
        }

    def reset_baseline(self):
        """Reset calibration."""
        self.state = self.state._replace(
            baseline_variance=jnp.ones((self.num_routers, self.num_bands)) * 2.0,
            baseline_samples=0,
        )
        logger.info("Baseline reset")


async def main():
    """Demo with synthetic data."""
    logging.basicConfig(level=logging.INFO)

    processor = SignalProcessor(num_routers=10, num_bands=2)

    logger.info("Processing 200 synthetic RSSI samples...")

    for i in range(200):
        # Synthetic RSSI: -60 dBm ± 5 dBm
        rssi_data = np.random.normal(-60, 5, size=(10, 2)).astype(np.float32)
        result = processor.process_rssi(rssi_data)

        if i == 50:
            if result["csi_virtual"] is not None:
                logger.info(f"✓ Frame {i}: CSI shape {result['csi_virtual'].shape}, "
                           f"mean variance {result['csi_virtual'].mean():.4f}")

    stats = processor.get_stats()
    logger.info(f"Final stats: {stats}")
    logger.info("✓ Signal Processor test completed")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
