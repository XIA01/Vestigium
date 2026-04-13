"""
VESTIGIUM WebSocket Server - Real-time visualization streaming
Async websockets server, no blocking operations.
"""

import asyncio
import json
import base64
import logging
from typing import Dict, Any
from io import BytesIO
import websockets
from websockets.server import WebSocketServerProtocol

try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)


class WebSocketServer:
    """
    Async WebSocket server for real-time VESTIGIUM visualization.

    Streams clusters, heatmap (PNG base64), and statistics at 30 FPS.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        """
        Initialize WebSocket server.

        Args:
            host: Bind address
            port: Bind port
        """
        self.host = host
        self.port = port
        self.clients = set()
        self.queue = asyncio.Queue(maxsize=2)  # Buffer latest 2 frames

        logger.info(f"WebSocketServer initialized on ws://{host}:{port}")

    async def start(self):
        """Start the server (non-blocking)."""
        async with websockets.serve(self.handler, self.host, self.port):
            logger.info(f"✓ WebSocket server running on ws://{self.host}:{self.port}")
            await asyncio.Event().wait()  # Run forever

    async def handler(self, websocket: WebSocketServerProtocol, path: str):
        """Handle incoming WebSocket connection."""
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}. Total: {len(self.clients)}")

        try:
            async for message in websocket:
                # Echo any client messages (for ping/pong or commands)
                if message == "ping":
                    await websocket.send("pong")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            logger.info(f"Client disconnected. Total: {len(self.clients)}")

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

            # Broadcast to all clients (with timeout)
            disconnected = []
            for client in self.clients:
                try:
                    await asyncio.wait_for(client.send(message), timeout=1.0)
                except Exception as e:
                    disconnected.append(client)
                    logger.debug(f"Failed to send to client: {e}")

            # Remove disconnected clients
            for client in disconnected:
                self.clients.discard(client)

        except Exception as e:
            logger.error(f"Error broadcasting frame: {e}")

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
        """Queue a frame for broadcast."""
        try:
            self.queue.put_nowait(frame_data)
        except asyncio.QueueFull:
            # Drop old frame, add new one
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.queue.put_nowait(frame_data)

    async def broadcast_loop(self):
        """
        Background task that broadcasts queued frames.
        Run this concurrently with the server.
        """
        while True:
            try:
                frame_data = await self.queue.get()
                await self.broadcast_frame(frame_data)
            except Exception as e:
                logger.error(f"Error in broadcast loop: {e}")
            await asyncio.sleep(0.001)  # Minimal sleep


async def main():
    """Demo: start server, wait for clients."""
    logging.basicConfig(level=logging.INFO)

    server = WebSocketServer()

    # Start server and broadcast loop
    server_task = asyncio.create_task(server.start())
    broadcast_task = asyncio.create_task(server.broadcast_loop())

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

    await asyncio.gather(server_task, broadcast_task, sim_task)


if __name__ == "__main__":
    asyncio.run(main())
