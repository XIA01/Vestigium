"""
VESTIGIUM WiFi Scanner - Real RSSI ingestion via iw / /proc/net/wireless
No mocks. Real data from Linux wireless interfaces.
"""

import asyncio
import re
import logging
from typing import AsyncGenerator, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class WifiScanner:
    """
    Real WiFi RSSI scanner using Linux iw command and /proc/net/wireless.

    Async generator that yields (num_aps, 2) arrays:
    - Column 0: 2.4GHz RSSI (dBm)
    - Column 1: 5GHz RSSI (dBm)
    """

    def __init__(self, interface: str = "wlan0", max_aps: int = 153):
        """
        Initialize WiFi scanner.

        Args:
            interface: WiFi interface name (e.g., "wlan0")
            max_aps: Maximum number of APs to track
        """
        self.interface = interface
        self.max_aps = max_aps
        self.ap_registry: Dict[str, int] = {}  # BSSID → index
        self.next_ap_idx = 0
        self.last_rssi: Dict[Tuple[str, str], float] = {}  # (BSSID, band) → RSSI

        logger.info(f"WifiScanner initialized on {interface}, max {max_aps} APs")

    async def stream_rssi(self) -> AsyncGenerator[np.ndarray, None]:
        """
        Async generator yielding RSSI matrices.

        Yields:
            np.ndarray of shape (num_aps, 2) with RSSI values in dBm
        """
        try:
            # Try high-speed iw scanning first
            async for rssi_matrix in self._scan_iw_fast():
                yield rssi_matrix
        except Exception as e:
            logger.warning(f"iw scanning failed: {e}, falling back to /proc/net/wireless")
            # Fallback: read /proc/net/wireless (single AP, continuous)
            async for rssi_matrix in self._scan_procfs():
                yield rssi_matrix

    async def _scan_iw_fast(self) -> AsyncGenerator[np.ndarray, None]:
        """
        Scan using 'iw dev <interface> scan dump' with no sleep.
        Parses BSS entries for BSSID, signal, frequency.
        """
        while True:
            try:
                # Run iw scan dump
                proc = await asyncio.create_subprocess_exec(
                    "iw", "dev", self.interface, "scan", "dump",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()

                if proc.returncode != 0:
                    raise RuntimeError(f"iw error: {stderr.decode()}")

                # Parse output
                self._parse_iw_output(stdout.decode())

                # Build matrix
                rssi_matrix = self._build_rssi_matrix()
                if rssi_matrix is not None:
                    yield rssi_matrix

                # No sleep—go immediately to next scan
                await asyncio.sleep(0.01)  # Minimal delay to avoid CPU saturation

            except Exception as e:
                logger.error(f"iw scan error: {e}")
                await asyncio.sleep(1)
                continue

    def _parse_iw_output(self, output: str) -> None:
        """
        Parse 'iw scan dump' output.
        Format:
        BSS 00:11:22:33:44:55 (on wlan0)
            signal: -42 dBm
            freq: 2437
        ...
        """
        lines = output.split("\n")
        current_bssid = None
        current_band = None

        for line in lines:
            # Detect BSS (new AP)
            bss_match = re.match(r"^BSS ([0-9a-fA-F:]{17})", line)
            if bss_match:
                current_bssid = bss_match.group(1).lower()
                # Register AP if new
                if current_bssid not in self.ap_registry:
                    if self.next_ap_idx < self.max_aps:
                        self.ap_registry[current_bssid] = self.next_ap_idx
                        self.next_ap_idx += 1
                        logger.debug(f"Registered AP {current_bssid} at index {self.ap_registry[current_bssid]}")

            # Detect signal strength
            signal_match = re.search(r"signal: (-?\d+) dBm", line)
            if signal_match and current_bssid is not None:
                rssi = float(signal_match.group(1))
                self.last_rssi[(current_bssid, current_band or "unknown")] = rssi

            # Detect frequency (determine band)
            freq_match = re.search(r"freq: (\d+)", line)
            if freq_match and current_bssid is not None:
                freq = int(freq_match.group(1))
                if freq < 3000:  # 2.4 GHz
                    current_band = "2.4"
                elif freq < 6000:  # 5 GHz
                    current_band = "5"
                else:  # 6 GHz or other
                    current_band = "other"

    def _build_rssi_matrix(self) -> Optional[np.ndarray]:
        """
        Build RSSI matrix from registry.
        Shape: (num_aps, 2) where col 0=2.4GHz, col 1=5GHz.
        Missing values: -100 dBm (out of range).
        """
        if not self.ap_registry:
            return None

        num_aps = len(self.ap_registry)
        matrix = np.ones((num_aps, 2)) * -100  # Default: out of range

        for (bssid, band), rssi in self.last_rssi.items():
            if bssid not in self.ap_registry:
                continue
            ap_idx = self.ap_registry[bssid]
            if band == "2.4":
                matrix[ap_idx, 0] = rssi
            elif band == "5":
                matrix[ap_idx, 1] = rssi

        return matrix

    async def _scan_procfs(self) -> AsyncGenerator[np.ndarray, None]:
        """
        Fallback: read /proc/net/wireless for connected AP only.
        Continuous loop at ~100 Hz.

        Format:
        Inter-| sta-|   Quality        |   Discarded Packets   |
        face  | tus | link level noise | nwid crypt frag retry misc
        wlan0: 0000   42. -42. -100     0     0     0     0     0
        """
        freq = 100  # Target 100 Hz sampling
        interval = 1.0 / freq

        while True:
            try:
                with open("/proc/net/wireless", "r") as f:
                    lines = f.readlines()

                # Find line for our interface
                for line in lines:
                    if line.startswith(self.interface):
                        # Parse: wlan0: 0000   42. -42. -100     0     0     0     0     0
                        parts = line.split()
                        if len(parts) >= 5:
                            link = float(parts[2].rstrip("."))  # Link quality
                            level = float(parts[3])  # Signal level (dBm)
                            noise = float(parts[4])  # Noise level (dBm)

                            # Assume this is a 2.4 GHz connection (could detect from wpa_cli)
                            matrix = np.ones((1, 2)) * -100
                            matrix[0, 0] = level  # Put in 2.4 GHz column

                            yield matrix
                        break

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"procfs scan error: {e}")
                await asyncio.sleep(1)
                continue

    async def get_ap_positions(self) -> Dict[str, Tuple[float, float]]:
        """
        Get estimated positions of APs.
        In a real system, this comes from trilateration or ML model.
        For now, return evenly-spaced grid positions.

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
    """Demo: scan WiFi and print RSSI table."""
    logging.basicConfig(level=logging.INFO)

    scanner = WifiScanner()
    count = 0

    try:
        async for rssi_matrix in scanner.stream_rssi():
            count += 1
            if count % 10 == 0:  # Print every 10 scans
                print(f"\n=== Scan {count} ===")
                print(f"Active APs: {len(scanner.ap_registry)}")
                print(f"Shape: {rssi_matrix.shape}")
                print(f"2.4GHz min/max: {rssi_matrix[:, 0].min():.0f} / {rssi_matrix[:, 0].max():.0f} dBm")
                print(f"5GHz min/max: {rssi_matrix[:, 1].min():.0f} / {rssi_matrix[:, 1].max():.0f} dBm")
                print(f"APs registry: {scanner.ap_registry}")

            if count >= 100:  # Run 100 scans then stop
                break

    except KeyboardInterrupt:
        logger.info("Scanner stopped")


if __name__ == "__main__":
    asyncio.run(main())
