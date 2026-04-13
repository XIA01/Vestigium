[app]

title = VESTIGIUM WiFi Scanner
package.name = vestigium_wifi
package.domain = vestigium

source.dir = .
source.include_exts = py

version = 0.1

# Minimal requirements - no JNI dependencies
requirements = python3

# Add requests via pip, not p4a
p4a.requires = requests

orientation = portrait
fullscreen = 0

android.permissions = CHANGE_WIFI_STATE,ACCESS_WIFI_STATE,ACCESS_COARSE_LOCATION,ACCESS_FINE_LOCATION,INTERNET

android.api = 31
android.minapi = 21
android.ndk = 25c
android.accept_sdk_license = True

log_level = 2
warn_on_root = 1

# Service bootstrap - no UI
p4a_bootstrap = service
p4a.setup_dir = .buildozer/android/platform/build-{arch}/build/other_builds

# Exclude JNI
p4a_private_storage = True
