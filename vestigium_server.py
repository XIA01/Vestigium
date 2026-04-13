#!/usr/bin/env python3
"""
VESTIGIUM Server - Entry point
Starts FastAPI server + processing pipeline
"""

import asyncio
import sys
import logging

sys.path.insert(0, '/media/latin/60FD21291B249B8D8/Programacion/HP')

from src.main import VestigiumSystem
from src.utils import setup_logging
from src.visualization import WebSocketServer

# Setup logging
setup_logging(level="INFO", log_file="logs/vestigium.log")
logger = logging.getLogger(__name__)


async def run_pipeline(system):
    """Run the processing pipeline in background"""
    await system.processing_loop()


def get_app():
    """Create and return FastAPI app for uvicorn"""
    system = VestigiumSystem(simulate=False)
    app = system.ws_server.get_app()

    # Store system reference in app state for access in routes
    app.state.system = system

    # Create background task for processing
    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting VESTIGIUM processing pipeline...")
        asyncio.create_task(run_pipeline(system))

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down VESTIGIUM...")
        system.shutdown()

    return app


if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 60)
    logger.info("VESTIGIUM - FastAPI WebSocket Server")
    logger.info("=" * 60)

    uvicorn.run(
        "vestigium_server:get_app",
        host="0.0.0.0",
        port=5000,
        reload=False,
        log_level="info",
        factory=True,
    )
