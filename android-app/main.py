"""
VESTIGIUM WiFi Scanner Service
Scans WiFi networks and sends data to server in real-time
No UI - runs as background service
"""

import json
import subprocess
import requests
import threading
import time
import os
import sys
from datetime import datetime

class WifiScannerService:
    """WiFi scanning service for Android"""

    def __init__(self, server_ip="192.168.1.247", server_port=5000):
        self.server_url = f"http://{server_ip}:{server_port}/api/wifi"
        self.scanning = True
        self.scan_count = 0
        self.log_file = "/sdcard/vestigium_wifi.log"

    def log(self, message: str):
        """Log to file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_msg + '\n')
        except:
            pass

    def parse_wifi_output(self, output: str) -> list:
        """Parse WiFi scan results from 'cmd wifi list-scan-results'"""
        networks = []
        lines = output.split('\n')

        for line in lines:
            if not line or 'BSSID' in line or 'Frequency' in line:
                continue

            try:
                parts = line.split()
                if len(parts) < 3:
                    continue

                bssid = parts[0]
                freq = int(parts[1])
                rssi = int(parts[2])
                ssid = ' '.join(parts[4:]) if len(parts) > 4 else '(hidden)'

                band = '2.4GHz' if freq < 3000 else '5GHz' if freq < 6000 else '6GHz'

                networks.append({
                    'bssid': bssid,
                    'ssid': ssid,
                    'rssi': rssi,
                    'frequency': freq,
                    'band': band
                })
            except (ValueError, IndexError):
                continue

        return networks

    def send_to_server(self, networks: list):
        """Send WiFi data to VESTIGIUM server"""
        try:
            # Convert to RSSI matrix format
            ap_registry = {}
            matrix = []

            for net in networks:
                bssid = net['bssid']
                if bssid not in ap_registry:
                    ap_registry[bssid] = len(ap_registry)

                idx = ap_registry[bssid]
                band = 0 if net['band'] == '2.4GHz' else 1

                # Ensure matrix is large enough
                while len(matrix) <= idx:
                    matrix.append([-100, -100])

                matrix[idx][band] = net['rssi']

            # Pad to 153 APs
            while len(matrix) < 153:
                matrix.append([-100, -100])

            # Send JSON
            data = {
                'timestamp': datetime.now().isoformat(),
                'aps': len(networks),
                'matrix': matrix[:153]
            }

            response = requests.post(self.server_url, json=data, timeout=2)
            return response.status_code == 200
        except Exception as e:
            self.log(f"Server error: {str(e)}")
            return False

    def scan_loop(self):
        """Background WiFi scanning loop"""
        self.log("WiFi Scanner started")

        while self.scanning:
            try:
                # Get WiFi scan results
                result = subprocess.run(
                    ['cmd', 'wifi', 'list-scan-results'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if result.returncode == 0:
                    networks = self.parse_wifi_output(result.stdout)
                    self.scan_count += 1

                    # Send to server
                    success = self.send_to_server(networks)

                    status = "OK" if success else "OFFLINE"
                    self.log(f"Scan #{self.scan_count}: {len(networks)} APs - {status}")
                else:
                    self.log(f"WiFi scan failed: {result.stderr}")

                # Wait before next scan (500ms)
                time.sleep(0.5)

            except Exception as e:
                self.log(f"Error: {str(e)}")
                time.sleep(1)

    def start(self):
        """Start scanning in background thread"""
        threading.Thread(target=self.scan_loop, daemon=True).start()

    def stop(self):
        """Stop scanning"""
        self.scanning = False


if __name__ == '__main__':
    # Parse command line arguments for server IP
    server_ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.247"
    server_port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

    service = WifiScannerService(server_ip, server_port)
    service.start()

    # Keep running
    try:
        while service.scanning:
            time.sleep(1)
    except KeyboardInterrupt:
        service.log("Scanner stopped")
        service.stop()
