[app]

title = VESTIGIUM WiFi Scanner
package.name = vestigium_wifi
package.domain = vestigium

source.dir = .
source.include_exts = py

version = 0.1

requirements = python3,requests

orientation = portrait
fullscreen = 0

android.permissions = CHANGE_WIFI_STATE,ACCESS_WIFI_STATE,ACCESS_COARSE_LOCATION,ACCESS_FINE_LOCATION,INTERNET

android.api = 31
android.minapi = 21
android.ndk = 25c
android.accept_sdk_license = True

log_level = 2
warn_on_root = 1

# Service bootstrap
p4a_bootstrap = service
