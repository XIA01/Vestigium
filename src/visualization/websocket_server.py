"""
VESTIGIUM WebSocket Server - FastAPI Edition
Clean, simple, production-ready WebSocket + HTTP server
"""

import json
import base64
import logging
from typing import Dict, Any
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)


class WebSocketServer:
    """FastAPI-based WebSocket server for real-time visualization"""

    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        """Initialize FastAPI server"""
        self.host = host
        self.port = port
        self.app = FastAPI(title="VESTIGIUM")
        self.clients = set()

        # Setup routes
        self._setup_routes()

        logger.info(f"WebSocketServer initialized (FastAPI)")
        logger.info(f"  HTTP: http://{host}:{port}/")
        logger.info(f"  WebSocket: ws://{host}:{port}/ws")

    def _setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/")
        async def get_root():
            """Serve HTML frontend"""
            frontend_path = Path(__file__).parent / "frontend" / "index.html"
            if frontend_path.exists():
                return FileResponse(frontend_path, media_type="text/html")
            return {"error": "Frontend not found"}

        @self.app.get("/favicon.ico")
        async def favicon():
            """Dummy favicon to prevent 404 errors"""
            return {"error": "Not found"}

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time frame streaming"""
            await websocket.accept()
            self.clients.add(websocket)
            client_addr = websocket.client
            logger.info(f"✓ WebSocket client connected: {client_addr}. Total: {len(self.clients)}")

            try:
                while True:
                    # Keep connection alive, receiving pings
                    data = await websocket.receive_text()
                    if data == "ping":
                        await websocket.send_text("pong")
            except WebSocketDisconnect:
                self.clients.discard(websocket)
                logger.info(f"WebSocket client disconnected. Total: {len(self.clients)}")
            except Exception as e:
                self.clients.discard(websocket)
                logger.debug(f"WebSocket error: {e}")

    async def start(self):
        """Start FastAPI server (called by uvicorn in main.py)"""
        logger.info(f"✓ FastAPI server ready")
        # Note: This is called by uvicorn, not directly
        pass

    async def broadcast_frame(self, frame_data: Dict[str, Any]):
        """Broadcast frame to all connected WebSocket clients"""
        if not self.clients:
            return

        try:
            # Encode images
            heatmap_b64 = self._encode_heatmap(frame_data.get("heatmap"))
            obstacle_b64 = self._encode_heatmap(frame_data.get("obstacle_map"))

            # Build JSON payload
            payload = {
                "clusters": frame_data.get("clusters", []),
                "heatmap": heatmap_b64,
                "obstacle_map": obstacle_b64,
                "stats": frame_data.get("stats", {}),
                "timestamp": frame_data.get("timestamp", 0),
            }

            message = json.dumps(payload)

            # Send to all clients
            disconnected = []
            for client in list(self.clients):
                try:
                    await client.send_text(message)
                except Exception:
                    disconnected.append(client)

            # Remove disconnected clients
            for client in disconnected:
                self.clients.discard(client)

        except Exception as e:
            logger.debug(f"Broadcast error: {e}")

    async def send_frame(self, frame_data: Dict[str, Any]):
        """Send frame (alias for broadcast_frame)"""
        await self.broadcast_frame(frame_data)

    async def broadcast_loop(self):
        """Broadcast loop (for compatibility, FastAPI handles async natively)"""
        import asyncio
        await asyncio.Event().wait()

    @staticmethod
    def _encode_heatmap(heatmap) -> str:
        """Encode heatmap numpy array to PNG base64"""
        if heatmap is None or Image is None:
            return ""

        try:
            import numpy as np

            # Ensure uint8
            if heatmap.dtype != np.uint8:
                heatmap = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)

            # Convert to RGB
            if len(heatmap.shape) == 2:
                rgb = np.stack([heatmap, heatmap, heatmap], axis=2)
            else:
                rgb = heatmap

            # Encode to PNG
            img = Image.fromarray(rgb, mode="RGB")
            buf = BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)

            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{b64}"

        except Exception as e:
            logger.error(f"Failed to encode heatmap: {e}")
            return ""

    def get_app(self):
        """Return FastAPI app instance for uvicorn"""
        return self.app
