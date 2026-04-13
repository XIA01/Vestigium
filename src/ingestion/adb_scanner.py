"""
VESTIGIUM WiFi Scanner via ADB - Real RSSI from Android device
Connects to ZTE Blade A71 or compatible Android phone via USB
Extracts live WiFi scan results with RSSI, frequency, SSID
Non-blocking background update with cached results.
"""

import asyncio
import subprocess
import re
import logging
from typing import AsyncGenerator, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
import threading
import time

logger = logging.getLogger(__name__)


class ADBWifiScanner:
    """
    Real WiFi RSSI scanner via ADB from Android device.

    Async generator that yields (num_aps, 2) arrays:
    - Column 0: 2.4GHz RSSI (dBm)
    - Column 1: 5GHz RSSI (dBm)

    Uses background thread for ADB calls to avoid blocking async loop.
    """

    def __init__(self, max_aps: int = 153, poll_interval_ms: int = 500):
        """
        Initialize ADB WiFi scanner.

        Args:
            max_aps: Maximum number of APs to track
            poll_interval_ms: How often to update WiFi scans from phone
        """
        self.max_aps = max_aps
        self.poll_interval = poll_interval_ms / 1000.0
        self.ap_registry: Dict[str, int] = {}  # BSSID → index
        self.next_ap_idx = 0
        self.last_rssi: Dict[Tuple[str, str], float] = {}  # (BSSID, band) → RSSI

        # Background thread for ADB scanning
        self.adb_thread_running = True
        self.adb_thread = threading.Thread(target=self._adb_background_worker, daemon=True)
        self.adb_thread.start()

        logger.info(f"ADBWifiScanner initialized, max {max_aps} APs, polling in background")

    async def stream_rssi(self) -> AsyncGenerator[np.ndarray, None]:
        """
        Async generator yielding RSSI matrices from cached phone WiFi scans.
        Background thread keeps cache updated via ADB.

        Yields:
            np.ndarray of shape (num_aps, 2) with RSSI values in dBm
        """
        while self.adb_thread_running:
            try:
                # Build matrix from cached data (no blocking)
                rssi_matrix = self._build_rssi_matrix()
                if rssi_matrix is not None:
                    yield rssi_matrix

                await asyncio.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"ADB stream error: {e}")
                await asyncio.sleep(0.5)
                continue

    def _adb_background_worker(self) -> None:
        """
        Background thread worker that continuously updates WiFi data via ADB.
        Non-blocking to the async event loop.
        """
        logger.info("ADB background worker started")
        scan_count = 0

        while self.adb_thread_running:
            try:
                # Execute ADB command (blocking, but in background thread)
                output = self._run_adb_blocking()
                if output:
                    self._parse_wifi_output(output)
                    scan_count += 1
                    if scan_count == 1:
                        logger.info(f"✓ First WiFi scan complete: {len(self.ap_registry)} APs detected")

                # Sleep between scans (no sleep on first iteration)
                if scan_count > 0:
                    time.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"ADB background worker error: {e}")
                time.sleep(1)

    def _run_adb_blocking(self) -> str:
        """
        Run ADB command synchronously (called from background thread, so blocking is OK).
        """
        try:
            result = subprocess.run(
                ["adb", "shell", "cmd", "wifi", "list-scan-results"],
                capture_output=True,
                timeout=2,
                text=True
            )
            if result.returncode == 0:
                return result.stdout
        except subprocess.TimeoutExpired:
            logger.warning("ADB timeout, trying dumpsys...")
        except Exception as e:
            logger.warning(f"ADB error: {e}")

        # Fallback to dumpsys
        try:
            result = subprocess.run(
                ["adb", "shell", "dumpsys", "wifi"],
                capture_output=True,
                timeout=3,
                text=True
            )
            if result.returncode == 0:
                return result.stdout
        except Exception as e:
            logger.debug(f"Dumpsys error: {e}")

        return ""

    def _parse_wifi_output(self, output: str) -> None:
        """
        Parse WiFi scan results.
        Format from `cmd wifi list-scan-results`:
            BSSID              Frequency      RSSI           Age(sec)     SSID
          72:df:8d:2d:a7:3c       2457        -43            >1000.0
          30:df:8d:2d:a7:39       2457        -45            >1000.0    Manbru
        """
        lines = output.split("\n")

        for line in lines:
            line = line.strip()
            if not line or "BSSID" in line or "Frequency" in line:
                continue

            # Parse: BSSID, Frequency, RSSI, Age, SSID
            parts = line.split()
            if len(parts) < 3:
                continue

            try:
                bssid = parts[0].lower()
                freq = int(parts[1])
                rssi = int(parts[2])

                # Determine band from frequency
                if freq < 3000:  # 2.4 GHz
                    band = "2.4"
                elif freq < 6000:  # 5 GHz
                    band = "5"
                else:  # 6 GHz or other
                    band = "other"

                # Register AP if new
                if bssid not in self.ap_registry:
                    if self.next_ap_idx < self.max_aps:
                        self.ap_registry[bssid] = self.next_ap_idx
                        self.next_ap_idx += 1
                        ssid = " ".join(parts[4:]) if len(parts) > 4 else "(hidden)"
                        logger.debug(f"Registered AP {bssid} ({ssid}) at index {self.ap_registry[bssid]}")

                # Store RSSI
                self.last_rssi[(bssid, band)] = float(rssi)

            except (ValueError, IndexError) as e:
                logger.debug(f"Parse error: {e}, line: {line}")
                continue

    def _build_rssi_matrix(self) -> Optional[np.ndarray]:
        """
        Build RSSI matrix from registry, padded to max_aps.
        Shape: (max_aps, 2) where col 0=2.4GHz, col 1=5GHz.
        Missing values: -100 dBm (out of range).
        """
        if not self.ap_registry:
            return None

        # Always return max_aps rows, padding with -100 for undetected APs
        matrix = np.ones((self.max_aps, 2)) * -100  # Default: out of range

        for (bssid, band), rssi in self.last_rssi.items():
            if bssid not in self.ap_registry:
                continue
            ap_idx = self.ap_registry[bssid]
            if band == "2.4":
                matrix[ap_idx, 0] = rssi
            elif band == "5":
                matrix[ap_idx, 1] = rssi

        return matrix

    async def get_ap_positions(self) -> Dict[str, Tuple[float, float]]:
        """
        Get estimated positions of APs (grid layout for now).

        Returns:
            {BSSID: (x, y)} in meters
        """
        positions = {}
        grid_size = int(np.ceil(np.sqrt(len(self.ap_registry))))

        for bssid, idx in self.ap_registry.items():
            row = idx // grid_size
            col = idx % grid_size
            x = (col - grid_size / 2) * 10  # 10m spacing
            y = (row - grid_size / 2) * 10
            positions[bssid] = (x, y)

        return positions


async def main():
    """Demo: scan WiFi via ADB and print RSSI table."""
    logging.basicConfig(level=logging.INFO)

    scanner = ADBWifiScanner()
    count = 0

    try:
        async for rssi_matrix in scanner.stream_rssi():
            count += 1
            if count % 5 == 0:  # Print every 5 scans
                print(f"\n=== Scan {count} ===")
                print(f"Active APs: {len(scanner.ap_registry)}")
                print(f"Shape: {rssi_matrix.shape}")
                print(f"2.4GHz min/max: {rssi_matrix[:, 0].min():.0f} / {rssi_matrix[:, 0].max():.0f} dBm")
                print(f"5GHz min/max: {rssi_matrix[:, 1].min():.0f} / {rssi_matrix[:, 1].max():.0f} dBm")

            if count >= 50:  # Run 50 scans then stop
                break

    except KeyboardInterrupt:
        logger.info("Scanner stopped")


if __name__ == "__main__":
    asyncio.run(main())
