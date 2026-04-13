"""
VESTIGIUM WebSocket Server - Simplified version
Pure websockets library + simple HTTP server
"""

import asyncio
import json
import base64
import logging
from typing import Dict, Any
from io import BytesIO
from pathlib import Path
import http.server
import threading

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)


class SimpleHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """Serve HTML frontend"""

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            frontend_path = Path(__file__).parent / "frontend" / "index.html"
            if frontend_path.exists():
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()
                with open(frontend_path, 'rb') as f:
                    self.wfile.write(f.read())
                return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        # Suppress default logging
        pass


class WebSocketServer:
    """
    WebSocket server using pure websockets library.
    HTTP served via simple threading server.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        if not HAS_WEBSOCKETS:
            raise RuntimeError("websockets not installed. Run: pip install websockets")

        self.host = host
        self.port = port
        self.ws_port = port
        self.http_port = port + 1000  # Use different port for HTTP
        self.clients = set()

        logger.info(f"WebSocketServer initialized")
        logger.info(f"  HTTP: http://{host}:{self.http_port}/")
        logger.info(f"  WebSocket: ws://{host}:{self.ws_port}")

    async def start(self):
        """Start WebSocket server"""
        # Start HTTP server in background thread
        self._start_http_server()

        # Start WebSocket server
        async with websockets.serve(self.handle_client, self.host, self.ws_port):
            logger.info(f"✓ WebSocket server listening on ws://{self.host}:{self.ws_port}")
            await asyncio.Event().wait()

    def _start_http_server(self):
        """Start HTTP server in background thread"""
        handler = SimpleHTTPHandler
        server = http.server.HTTPServer(("0.0.0.0", self.http_port), handler)

        def run_server():
            logger.info(f"✓ HTTP server listening on http://0.0.0.0:{self.http_port}")
            server.serve_forever()

        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()

    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection"""
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        logger.info(f"WebSocket client connected: {client_addr}. Total: {len(self.clients)}")

        try:
            async for message in websocket:
                if message == "ping":
                    await websocket.send("pong")
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.debug(f"WebSocket error: {e}")
        finally:
            self.clients.discard(websocket)
            logger.info(f"WebSocket client disconnected. Total: {len(self.clients)}")

    async def broadcast_frame(self, frame_data: Dict[str, Any]):
        """Broadcast frame to all connected clients"""
        if not self.clients:
            return

        try:
            heatmap_b64 = self._encode_heatmap(frame_data.get("heatmap"))
            obstacle_b64 = self._encode_heatmap(frame_data.get("obstacle_map"))

            payload = {
                "clusters": frame_data.get("clusters", []),
                "heatmap": heatmap_b64,
                "obstacle_map": obstacle_b64,
                "stats": frame_data.get("stats", {}),
                "timestamp": frame_data.get("timestamp", 0),
            }

            message = json.dumps(payload)

            # Broadcast
            disconnected = []
            for client in list(self.clients):
                try:
                    await asyncio.wait_for(client.send(message), timeout=1.0)
                except Exception as e:
                    disconnected.append(client)
                    logger.debug(f"Failed to send to client: {e}")

            for client in disconnected:
                self.clients.discard(client)

        except Exception as e:
            logger.error(f"Error broadcasting: {e}")

    async def send_frame(self, frame_data: Dict[str, Any]):
        """Queue frame for broadcast"""
        await self.broadcast_frame(frame_data)

    async def broadcast_loop(self):
        """Keep-alive loop"""
        await asyncio.Event().wait()

    @staticmethod
    def _encode_heatmap(heatmap) -> str:
        """Encode heatmap to PNG base64"""
        if heatmap is None or Image is None:
            return ""

        try:
            import numpy as np

            if heatmap.dtype != np.uint8:
                heatmap = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)

            if len(heatmap.shape) == 2:
                rgb = np.stack([heatmap, heatmap, heatmap], axis=2)
            else:
                rgb = heatmap

            img = Image.fromarray(rgb, mode="RGB")
            buf = BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)

            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{b64}"

        except Exception as e:
            logger.error(f"Failed to encode heatmap: {e}")
            return ""
