"""
VESTIGIUM Main - Full pipeline orchestration
Real WiFi ingestion → JAX processing → WebSocket visualization
All async, no blocking operations, 30 FPS target.
"""

import asyncio
import signal
import time
import numpy as np
import logging
from typing import Optional

from .ingestion import WifiScanner
from .backend import SignalProcessor, NeuromorphicEngine, SLAMTopological
from .visualization import WebSocketServer
from .utils import get_config, get_logger, setup_logging

logger = get_logger("vestigium.main")


class VestigiumSystem:
    """
    Complete VESTIGIUM pipeline.

    Orchestrates:
    1. WiFi RSSI ingestion (real or simulated)
    2. JAX GPU processing (signal → neuromorphic → SLAM)
    3. WebSocket visualization
    """

    def __init__(self, config_path: str = "config.yaml", simulate: bool = False):
        """
        Initialize VESTIGIUM system.

        Args:
            config_path: Path to config.yaml
            simulate: If True, use synthetic RSSI instead of real WiFi
        """
        self.config = get_config(config_path)
        self.simulate = simulate

        logger.info("=" * 60)
        logger.info("VESTIGIUM - WiFi Biomass Radar (JAX GPU)")
        logger.info("=" * 60)

        # Phase 1: Signal Processor
        self.signal_processor = SignalProcessor(
            num_routers=153,
            num_bands=2,
            window_size_ms=self.config.get("signal_processing.scintillation.window_size_ms", 100),
            sampling_rate_hz=self.config.get("hardware.polling.max_samples_per_second", 100),
        )
        logger.info("✓ Signal Processor (Phase 1) initialized")

        # Phase 2: Neuromorphic Engine
        self.neuromorphic = NeuromorphicEngine(
            num_neurons=self.config.get("neuromorphic.skan_network.num_neurons", 256),
            num_particles=self.config.get("neuromorphic.particle_filter.num_particles", 1000),
            spike_threshold=self.config.get("neuromorphic.skan_network.spike_threshold", 0.7),
            learning_rate=self.config.get("neuromorphic.skan_network.learning_rate", 0.001),
        )
        logger.info("✓ Neuromorphic Engine (Phase 2) initialized")

        # Phase 3: SLAM
        self.slam = SLAMTopological(
            map_width_m=self.config.get("slam.map.width_meters", 50),
            map_height_m=self.config.get("slam.map.height_meters", 50),
            resolution_cm=self.config.get("slam.map.resolution_cm", 10),
            decay_factor=self.config.get("slam.heatmap.decay_factor", 0.95),
        )
        logger.info("✓ SLAM Topological (Phase 3) initialized")

        # Phase 4: WebSocket Server
        self.ws_server = WebSocketServer(
            host=self.config.get("visualization.server.host", "0.0.0.0"),
            port=self.config.get("visualization.server.port", 5000),
        )
        logger.info("✓ WebSocket Server (Phase 4) initialized")

        # WiFi Scanner
        if simulate:
            self.wifi_scanner = None
            logger.info("✓ Using SIMULATED RSSI (no WiFi scanner)")
        else:
            self.wifi_scanner = WifiScanner()
            logger.info("✓ WiFi Scanner initialized")

        # Statistics
        self.frame_count = 0
        self.total_detections = 0
        self.start_time = time.time()
        self.shutdown_event = asyncio.Event()

    async def process_frame(self, rssi_data: np.ndarray) -> Optional[dict]:
        """
        Process one frame through all 4 phases.

        Args:
            rssi_data: np.ndarray (num_routers, num_bands)

        Returns:
            Visualization frame or None if not ready
        """
        self.frame_count += 1

        # Phase 1: Signal Processing (JAX jit)
        signal_result = self.signal_processor.process_rssi(rssi_data)

        if signal_result["csi_virtual"] is None:
            return None  # Buffer not full yet

        # Phase 2: Neuromorphic Engine (JAX jit + vmap)
        neuro_result = self.neuromorphic.process_csi_virtual(signal_result["csi_virtual"])

        # Phase 3: SLAM Topological (JAX jit)
        slam_result = self.slam.update_from_clusters(neuro_result["clusters"])

        # Update stats
        if neuro_result["clusters"]:
            self.total_detections += len(neuro_result["clusters"])

        # Build visualization frame
        frame = {
            "clusters": neuro_result["clusters"],
            "heatmap": self.slam.get_heatmap(),
            "obstacle_map": self.slam.get_obstacle_map(),
            "stats": {
                "fps": self._calculate_fps(),
                "detections": self.total_detections,
                "clusters": len(neuro_result["clusters"]),
            },
            "timestamp": time.time(),
        }

        return frame

    def _calculate_fps(self) -> float:
        """Calculate rolling FPS."""
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0

    async def ingestion_loop(self):
        """
        Async loop that pulls RSSI data (real or simulated).
        """
        logger.info("Starting ingestion loop...")

        if self.wifi_scanner:
            # Real WiFi data
            async for rssi_data in self.wifi_scanner.stream_rssi():
                if self.shutdown_event.is_set():
                    break

                yield rssi_data

        else:
            # Simulated data
            while not self.shutdown_event.is_set():
                # Synthetic RSSI with moving object pattern
                rssi_data = np.random.normal(-60, 5, size=(153, 2)).astype(np.float32)

                # Add simulated object
                if 50 < self.frame_count < 200:
                    angle = (self.frame_count - 50) * 0.05
                    x_perturb = np.sin(angle) * 3
                    y_perturb = np.cos(angle) * 3
                    rssi_data[:10, :] -= (x_perturb + y_perturb) * 2

                yield rssi_data
                await asyncio.sleep(0.01)  # ~100 Hz

    async def processing_loop(self):
        """
        Main processing loop: ingest → process 4 phases → visualize.
        """
        logger.info("Starting processing loop...")

        frame_count_at_last_log = 0

        async for rssi_data in self.ingestion_loop():
            if self.shutdown_event.is_set():
                break

            # Process frame through pipeline
            frame = await self.process_frame(rssi_data)

            if frame is not None:
                # Send to WebSocket
                await self.ws_server.send_frame(frame)

            # Log progress every 100 frames
            if self.frame_count % 100 == 0 and self.frame_count != frame_count_at_last_log:
                frame_count_at_last_log = self.frame_count
                fps = self._calculate_fps()
                logger.info(
                    f"Frame {self.frame_count}: "
                    f"FPS={fps:.1f}, "
                    f"Total detections={self.total_detections}, "
                    f"Occupancy={self.slam.get_stats()['mean_occupancy']:.3f}"
                )

    async def run(self):
        """
        Start all async tasks.
        """
        logger.info("\n" + "=" * 60)
        logger.info("Starting VESTIGIUM pipeline...")
        logger.info("=" * 60)

        # Create tasks
        server_task = asyncio.create_task(self.ws_server.start())
        processing_task = asyncio.create_task(self.processing_loop())

        # Wait indefinitely (only exit on shutdown_event)
        try:
            await asyncio.gather(server_task, processing_task)
        except asyncio.CancelledError:
            logger.info("Pipeline tasks cancelled")
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)

    def shutdown(self):
        """Graceful shutdown."""
        logger.info("\n" + "=" * 60)
        logger.info("Shutting down VESTIGIUM...")
        logger.info("=" * 60)

        self.shutdown_event.set()

        # Final stats
        stats = {
            "total_frames": self.frame_count,
            "total_detections": self.total_detections,
            "avg_fps": self._calculate_fps(),
            "signal_processor": self.signal_processor.get_stats(),
            "neuromorphic": self.neuromorphic.get_stats(),
            "slam": self.slam.get_stats(),
        }

        logger.info("\nFINAL STATISTICS:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        logger.info("=" * 60)
        logger.info("✓ VESTIGIUM shutdown complete")


async def main():
    """Main entry point."""
    setup_logging(level="INFO", log_file="logs/vestigium.log")

    # Initialize system
    system = VestigiumSystem(simulate=True)

    # Handle signals
    loop = asyncio.get_event_loop()

    def signal_handler():
        system.shutdown()
        loop.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await system.run()
    except KeyboardInterrupt:
        signal_handler()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        signal_handler()


if __name__ == "__main__":
    asyncio.run(main())
