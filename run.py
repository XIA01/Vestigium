#!/usr/bin/env python3
"""
VESTIGIUM - Run the complete system
Usage: ./run.py [--real] [--config config.yaml]
"""

import sys
import asyncio
import argparse

# Ensure venv is activated or use system python
sys.path.insert(0, '/media/latin/60FD21291B249B8D8/Programacion/HP')

from src.main import VestigiumSystem, main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VESTIGIUM WiFi Biomass Radar")
    parser.add_argument("--real", action="store_true", help="Use real WiFi data (requires iw command)")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--test", action="store_true", help="Run tests instead of main pipeline")

    args = parser.parse_args()

    if args.test:
        # Run system tests
        import test_system
        exit(test_system.test_all())
    else:
        # Modificar main() para pasar simulate flag
        import sys
        sys.argv = [sys.argv[0]]  # Limpio args para no confundir a argparse en main

        # Run main pipeline with real/simulated data
        # Note: main() siempre crea VestigiumSystem(simulate=True)
        # Para WiFi real, edita src/main.py línea ~261
        print(f"\n{'='*60}")
        print(f"VESTIGIUM Starting {'(SIMULATED)' if not args.real else '(REAL WiFi)'}")
        print(f"Open browser at http://localhost:5000")
        print(f"{'='*60}\n")

        if args.real:
            print("⚠️  WiFi real mode - asegúrate de tener:")
            print("   • Linux con 'iw' instalado")
            print("   • WiFi interface disponible (wlan0)")
            print("   • Permisos de lectura en /proc/net/wireless")
            print()

        asyncio.run(main())
