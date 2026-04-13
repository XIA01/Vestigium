"""
VESTIGIUM WebSocket Server - Real-time visualization streaming
HTTP + WebSocket server using aiohttp for full async support.
Serves HTML frontend + WebSocket data stream.
"""

import asyncio
import json
import base64
import logging
from typing import Dict, Any
from io import BytesIO
from pathlib import Path

try:
    from aiohttp import web
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    web = None

try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)


class WebSocketServer:
    """
    HTTP + WebSocket server for real-time VESTIGIUM visualization.

    Serves:
    - HTTP GET / → HTML frontend (index.html)
    - WebSocket /ws → Real-time frame streaming
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        """
        Initialize server.

        Args:
            host: Bind address
            port: Bind port
        """
        if not HAS_AIOHTTP:
            raise RuntimeError("aiohttp not installed. Run: pip install aiohttp")

        self.host = host
        self.port = port
        self.clients = set()
        self.app = None
        self.runner = None

        logger.info(f"WebSocketServer initialized on http://{host}:{port}")

    async def start(self):
        """Start the server."""
        self.app = web.Application()

        # Routes
        self.app.router.add_get("/", self.handle_index)
        self.app.router.add_get("/ws", self.handle_ws)

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()

        logger.info(f"✓ HTTP+WebSocket server running on http://{self.host}:{self.port}")
        logger.info(f"  Frontend: http://{self.host}:{self.port}/")
        logger.info(f"  WebSocket: ws://{self.host}:{self.port}/ws")

        # Keep running forever
        await asyncio.Event().wait()

    async def handle_index(self, request):
        """Serve HTML frontend."""
        frontend_path = Path(__file__).parent / "frontend" / "index.html"

        if not frontend_path.exists():
            return web.Response(text="Frontend not found", status=404)

        with open(frontend_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        return web.Response(text=html_content, content_type="text/html")

    async def handle_ws(self, request):
        """Handle WebSocket connection."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        client_addr = request.remote
        self.clients.add(ws)
        logger.info(f"WebSocket client connected: {client_addr}. Total: {len(self.clients)}")

        try:
            async for msg in ws.iter_any():
                # Handle incoming messages (ping/pong, etc)
                if isinstance(msg, str):
                    if msg == "ping":
                        try:
                            await ws.send_str("pong")
                        except Exception as e:
                            logger.debug(f"Error sending pong: {e}")
                            break
        except asyncio.CancelledError:
            logger.debug(f"WebSocket cancelled: {client_addr}")
        except Exception as e:
            logger.debug(f"WebSocket error ({client_addr}): {e}")
        finally:
            self.clients.discard(ws)
            logger.info(f"WebSocket client disconnected. Total: {len(self.clients)}")

        return ws

    async def broadcast_frame(self, frame_data: Dict[str, Any]):
        """
        Broadcast a visualization frame to all clients.

        Args:
            frame_data: {
                'clusters': [...],
                'heatmap': np.ndarray uint8,
                'obstacle_map': np.ndarray uint8,
                'stats': {...},
                'timestamp': float,
            }
        """
        if not self.clients:
            return

        try:
            # Encode heatmap to PNG base64
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

            # Broadcast to all clients
            disconnected = []
            for client in self.clients:
                try:
                    await client.send_str(message)
                except asyncio.TimeoutError:
                    disconnected.append(client)
                    logger.debug(f"Client send timeout")
                except ConnectionError:
                    disconnected.append(client)
                except RuntimeError:
                    # Client already closed
                    disconnected.append(client)
                except Exception as e:
                    disconnected.append(client)
                    logger.debug(f"Failed to send to client: {type(e).__name__}")

            # Remove disconnected clients
            for client in disconnected:
                self.clients.discard(client)

        except Exception as e:
            logger.debug(f"Error broadcasting frame: {type(e).__name__}: {e}")

    @staticmethod
    def _encode_heatmap(heatmap) -> str:
        """
        Encode heatmap to PNG base64.

        Args:
            heatmap: np.ndarray uint8 (H, W)

        Returns:
            Base64 string of PNG image
        """
        if heatmap is None or Image is None:
            return ""

        try:
            import numpy as np

            # Ensure uint8
            if heatmap.dtype != np.uint8:
                heatmap = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)

            # Convert to RGB (replicate grayscale to 3 channels)
            if len(heatmap.shape) == 2:
                rgb = np.stack([heatmap, heatmap, heatmap], axis=2)
            else:
                rgb = heatmap

            # Create PIL image
            img = Image.fromarray(rgb, mode="RGB")

            # Encode to PNG in memory
            buf = BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)

            # Base64 encode
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{b64}"

        except Exception as e:
            logger.error(f"Failed to encode heatmap: {e}")
            return ""

    async def send_frame(self, frame_data: Dict[str, Any]):
        """Queue a frame for broadcast (immediate in aiohttp)."""
        await self.broadcast_frame(frame_data)

    async def broadcast_loop(self):
        """
        Background task (not needed with aiohttp, but kept for compatibility).
        In aiohttp, broadcast_frame is already async.
        """
        # aiohttp handles async natively, no queue needed
        await asyncio.Event().wait()

    async def stop(self):
        """Stop the server gracefully."""
        if self.runner:
            await self.runner.cleanup()


async def main():
    """Demo: start server and simulate frames."""
    logging.basicConfig(level=logging.INFO)

    server = WebSocketServer()

    # Start server
    server_task = asyncio.create_task(server.start())

    # Simulate frame generation
    async def simulate_frames():
        import time
        import numpy as np

        for i in range(100):
            frame = {
                "clusters": [{"x": 10 * np.sin(i * 0.1), "y": 10 * np.cos(i * 0.1), "confidence": 0.7}],
                "heatmap": np.random.randint(0, 255, (100, 100), dtype=np.uint8),
                "obstacle_map": np.random.randint(0, 3, (100, 100), dtype=np.uint8),
                "stats": {"fps": 30, "detections": 1, "clusters": 1},
                "timestamp": time.time(),
            }
            await server.send_frame(frame)
            await asyncio.sleep(0.033)  # 30 FPS

    sim_task = asyncio.create_task(simulate_frames())

    await asyncio.gather(server_task, sim_task)


if __name__ == "__main__":
    asyncio.run(main())
