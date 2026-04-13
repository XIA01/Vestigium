"""
VESTIGIUM WiFi Scanner APK - Kivy App
Scans WiFi networks and sends data to server in real-time
"""

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.clock import Clock
from kivy.core.window import Window
import json
import subprocess
import requests
import threading
from datetime import datetime

Window.size = (480, 800)


class VestigiumWiFiApp(App):
    """Main WiFi scanning app for VESTIGIUM"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.server_url = "http://192.168.1.247:5000/api/wifi"  # Will be configurable
        self.scanning = False
        self.scan_count = 0
        self.last_aps = {}

    def build(self):
        """Build the UI"""
        self.title = "VESTIGIUM WiFi Scanner"

        # Main layout
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Server config
        config_layout = GridLayout(cols=2, size_hint_y=0.15)
        config_layout.add_widget(Label(text='Server IP:', size_hint_x=0.3))
        self.server_input = TextInput(
            text='192.168.1.247:5000',
            multiline=False,
            size_hint_x=0.7
        )
        config_layout.add_widget(self.server_input)
        main_layout.add_widget(config_layout)

        # Control buttons
        button_layout = BoxLayout(size_hint_y=0.1, spacing=10)
        self.scan_btn = Button(text='START SCANNING', background_color=(0.2, 0.6, 0.2, 1))
        self.scan_btn.bind(on_press=self.toggle_scanning)
        button_layout.add_widget(self.scan_btn)

        clear_btn = Button(text='CLEAR LOG', background_color=(0.6, 0.2, 0.2, 1))
        clear_btn.bind(on_press=self.clear_log)
        button_layout.add_widget(clear_btn)
        main_layout.add_widget(button_layout)

        # Status display
        self.status_label = Label(
            text='Ready. Configure server and click START.',
            size_hint_y=0.1,
            background_color=(0.1, 0.1, 0.1, 1)
        )
        main_layout.add_widget(self.status_label)

        # WiFi networks display
        scroll = ScrollView(size_hint_y=0.65)
        self.networks_layout = GridLayout(cols=1, spacing=5, size_hint_y=None)
        self.networks_layout.bind(minimum_height=self.networks_layout.setter('height'))
        scroll.add_widget(self.networks_layout)
        main_layout.add_widget(scroll)

        return main_layout

    def toggle_scanning(self, instance):
        """Start/stop WiFi scanning"""
        if self.scanning:
            self.scanning = False
            self.scan_btn.text = 'START SCANNING'
            self.scan_btn.background_color = (0.2, 0.6, 0.2, 1)
            self.update_status('Scanning stopped.')
        else:
            self.scanning = True
            self.scan_btn.text = 'STOP SCANNING'
            self.scan_btn.background_color = (0.6, 0.2, 0.2, 1)
            self.update_status('Starting WiFi scan...')
            # Start scanning in background
            threading.Thread(target=self.scan_loop, daemon=True).start()

    def scan_loop(self):
        """Background WiFi scanning loop"""
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

                    # Update UI
                    Clock.schedule_once(
                        lambda dt: self.update_networks_display(networks),
                        0
                    )

                    # Send to server
                    self.send_to_server(networks)

                    # Status update
                    Clock.schedule_once(
                        lambda dt: self.update_status(
                            f'Scan #{self.scan_count}: {len(networks)} APs found'
                        ),
                        0
                    )

                # Wait before next scan
                import time
                time.sleep(0.5)

            except Exception as e:
                Clock.schedule_once(
                    lambda dt: self.update_status(f'Error: {str(e)[:50]}'),
                    0
                )
                import time
                time.sleep(1)

    def parse_wifi_output(self, output: str) -> list:
        """Parse WiFi scan results"""
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

    def update_networks_display(self, networks: list):
        """Update the WiFi networks display"""
        self.networks_layout.clear_widgets()

        for net in networks[:20]:  # Show top 20
            label_text = (
                f"{net['ssid'][:30]:<30} | "
                f"{net['rssi']:>4} dBm | "
                f"{net['band']:<6}"
            )
            label = Label(
                text=label_text,
                size_hint_y=None,
                height=40,
                font_name='RobotoMono'
            )
            self.networks_layout.add_widget(label)

    def send_to_server(self, networks: list):
        """Send WiFi data to VESTIGIUM server"""
        try:
            server = self.server_input.text
            if ':' not in server:
                server = f"{server}:5000"

            url = f"http://{server}/api/wifi"

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

            requests.post(url, json=data, timeout=2)
        except Exception as e:
            pass  # Silently fail, server might be offline

    def update_status(self, message: str):
        """Update status label"""
        self.status_label.text = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"

    def clear_log(self, instance):
        """Clear the networks display"""
        self.networks_layout.clear_widgets()
        self.update_status('Log cleared.')


if __name__ == '__main__':
    VestigiumWiFiApp().run()
