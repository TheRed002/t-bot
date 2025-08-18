"""
Public WebSocket handlers for T-Bot web interface.

This module provides public WebSocket endpoints that don't require authentication,
for features like public market data and system status.
"""

import asyncio
import json
from datetime import datetime

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from src.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.websocket("")
async def public_websocket(websocket: WebSocket, token: str | None = Query(None)):
    """
    Public WebSocket endpoint for real-time updates.

    This endpoint provides:
    - Public market data
    - System status
    - General notifications
    - Authenticated features if token is provided
    """
    await websocket.accept()
    client_id = f"client_{id(websocket)}"
    is_authenticated = bool(token)

    logger.info(f"WebSocket connected: {client_id} (authenticated: {is_authenticated})")

    try:
        # Send welcome message
        await websocket.send_json(
            {
                "type": "welcome",
                "message": "Connected to T-Bot WebSocket"
                if is_authenticated
                else "Connected to T-Bot public stream",
                "timestamp": datetime.utcnow().isoformat(),
                "client_id": client_id,
                "authenticated": is_authenticated,
            }
        )

        # Keep connection alive with periodic ping
        while True:
            # Wait for client messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)

                # Handle different message types
                if message.get("type") == "ping":
                    await websocket.send_json(
                        {"type": "pong", "timestamp": datetime.utcnow().isoformat()}
                    )
                elif message.get("type") == "subscribe":
                    # Send confirmation
                    await websocket.send_json(
                        {
                            "type": "subscribed",
                            "channels": message.get("channels", []),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                else:
                    # Echo back for testing
                    await websocket.send_json(
                        {
                            "type": "echo",
                            "data": message,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

            except asyncio.TimeoutError:
                # Send heartbeat if no message received
                await websocket.send_json(
                    {"type": "heartbeat", "timestamp": datetime.utcnow().isoformat()}
                )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}")
        try:
            await websocket.close(code=1000)
        except:
            pass  # WebSocket might already be closed
