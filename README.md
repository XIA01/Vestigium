# VESTIGIUM WiFi Scanner APK

Real-time WiFi scanning app for VESTIGIUM biomass radar.

## Features

✅ Real-time WiFi network scanning (no cache)
✅ Sends RSSI data to VESTIGIUM server
✅ Shows live network list
✅ Configurable server IP
✅ Background scanning thread

## Installation

### Option 1: Download Pre-built APK (from GitHub Actions)

1. Go to https://github.com/YOUR_USERNAME/vestigium-wifi-apk/actions
2. Click the latest successful build
3. Download `vestigium-wifi-apk` artifact
4. Extract and install:
   ```bash
   adb install bin/vestigium_wifi-0.1-debug.apk
   ```

### Option 2: Build Locally

```bash
# Install buildozer and dependencies
pip install buildozer kivy requests cython

# Build APK
buildozer android debug

# Install
adb install bin/vestigium_wifi-0.1-debug.apk
```

## Usage

1. Launch the app on your phone
2. Enter server IP (default: 192.168.1.247:5000)
3. Click START SCANNING
4. App will scan WiFi and send data to server every 500ms

## Server Integration

The app sends WiFi data to `http://SERVER_IP:5000/api/wifi` as JSON:

```json
{
  "timestamp": "2026-04-12T23:45:00.123456",
  "aps": 8,
  "matrix": [
    [-43, -100],
    [-45, -100],
    ...
  ]
}
```

## Permissions

The app requires:
- `CHANGE_WIFI_STATE` - Enable/disable WiFi
- `ACCESS_WIFI_STATE` - Read WiFi status
- `ACCESS_COARSE_LOCATION` - WiFi scanning
- `ACCESS_FINE_LOCATION` - Precise WiFi data
- `INTERNET` - Send data to server

## Building with GitHub Actions

1. Push to GitHub
2. GitHub Actions automatically builds the APK
3. Download from Actions artifacts
4. Install with adb

No local Java/Android SDK needed!
