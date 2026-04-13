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
        # Run main pipeline
        asyncio.run(main())
