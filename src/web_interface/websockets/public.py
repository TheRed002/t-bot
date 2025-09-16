"""
Public WebSocket handlers for T-Bot web interface.

This module provides public WebSocket endpoints that don't require authentication,
for features like public market data and system status.
"""

import asyncio
import json
from datetime import datetime, timezone

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

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
    client_id = f"client_{id(websocket)}"
    is_authenticated = bool(token)

    try:
        # Accept connection with timeout
        await asyncio.wait_for(websocket.accept(), timeout=10.0)
        logger.info(f"WebSocket connected: {client_id} (authenticated: {is_authenticated})")

        # Send welcome message with timeout
        await websocket.send_json(
                {
                    "type": "welcome",
                    "message": (
                        "Connected to T-Bot WebSocket"
                        if is_authenticated
                        else "Connected to T-Bot public stream"
                    ),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "client_id": client_id,
                    "authenticated": is_authenticated,
                }
            )
    except asyncio.TimeoutError:
        logger.error(f"Timeout during WebSocket setup for {client_id}")
        return
    except Exception as e:
        logger.error(f"Error during WebSocket setup for {client_id}: {e}")
        return

    try:
        # Keep connection alive with periodic ping and proper message handling
        while True:
            try:
                # Wait for client messages with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)

                # Handle different message types with individual timeouts
                if message.get("type") == "ping":
                    try:
                        pong_message = {
                            "type": "pong",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        await asyncio.wait_for(websocket.send_json(pong_message), timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout sending pong to {client_id}")
                elif message.get("type") == "subscribe":
                    # Send confirmation with timeout
                    try:
                        await asyncio.wait_for(
                            websocket.send_json(
                                {
                                    "type": "subscribed",
                                    "channels": message.get("channels", []),
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                }
                            ),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout sending subscription response to {client_id}")
                else:
                    # Echo back for testing with timeout
                    try:
                        await asyncio.wait_for(
                            websocket.send_json(
                                {
                                    "type": "echo",
                                    "data": message,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                }
                            ),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout sending echo to {client_id}")

            except asyncio.TimeoutError:
                # Send heartbeat if no message received
                try:
                    heartbeat_message = {
                        "type": "heartbeat",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await asyncio.wait_for(websocket.send_json(heartbeat_message), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Failed to send heartbeat to {client_id}, disconnecting")
                    break
                except Exception as e:
                    logger.error(f"Error sending heartbeat to {client_id}: {e}")
                    break
            except json.JSONDecodeError:
                try:
                    await asyncio.wait_for(
                        websocket.send_json(
                            {
                                "type": "error",
                                "message": "Invalid JSON format",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        ),
                        timeout=3.0
                    )
                except Exception:
                    break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}")
    finally:
        # Proper cleanup with timeout
        try:
            # Only attempt to close if websocket is still connected
            if websocket.client_state not in (
                WebSocketState.DISCONNECTED,
                WebSocketState.CONNECTING,
            ):
                await asyncio.wait_for(websocket.close(code=1000), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout closing WebSocket for {client_id}")
        except (WebSocketDisconnect, ConnectionError, RuntimeError) as ws_error:
            # Expected WebSocket close errors - safe to ignore
            logger.debug(f"WebSocket already closed during cleanup: {ws_error}")
        except Exception as unexpected_error:
            # Unexpected errors should be logged for debugging
            logger.warning(f"Unexpected error during WebSocket cleanup: {unexpected_error}")
            # Don't re-raise - this is cleanup code
