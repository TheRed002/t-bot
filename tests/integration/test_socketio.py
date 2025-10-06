#!/usr/bin/env python3
"""
Integration test script for Socket.IO functionality.
Run this to verify Socket.IO is working properly between frontend and backend.
"""

import asyncio
import sys
from datetime import datetime

import socketio

# Create a Socket.IO client
sio = socketio.AsyncClient()


@sio.event
async def connect():
    print(f"✅ Connected to server at {datetime.now()}")

    # Try to authenticate
    await sio.emit("authenticate", {"token": "test-token-123"})


@sio.event
async def disconnect():
    print(f"❌ Disconnected from server at {datetime.now()}")


@sio.event
async def welcome(data):
    print(f"📨 Welcome message: {data}")


@sio.event
async def authenticated(data):
    print(f"🔐 Authentication response: {data}")

    if data.get("status") == "success":
        # Subscribe to channels
        await sio.emit("subscribe", {"channels": ["market_data", "bot_status", "portfolio"]})

        # Test ping
        await sio.emit("ping", {"timestamp": datetime.utcnow().isoformat()})

        # Request portfolio
        await sio.emit("get_portfolio", {})


@sio.event
async def auth_error(data):
    print(f"❌ Authentication error: {data}")


@sio.event
async def subscribed(data):
    print(f"📡 Subscribed to channels: {data}")


@sio.event
async def pong(data):
    print(f"🏓 Pong received: {data}")


@sio.event
async def market_data(data):
    print(f"📈 Market data update: {data.get('type', 'unknown')}")


@sio.event
async def bot_status(data):
    print(f"🤖 Bot status update: Active bots: {data.get('data', {}).get('active_bots', 0)}")


@sio.event
async def portfolio_update(data):
    print(f"💼 Portfolio update: Total value: ${data.get('data', {}).get('total_value', 0):.2f}")


@sio.event
async def portfolio_data(data):
    print(f"💰 Portfolio snapshot: Total value: ${data.get('total_value', 0):.2f}")


@sio.event
async def error(data):
    print(f"❌ Error: {data}")


async def main():
    """Main test function."""
    print("=" * 60)
    print("Socket.IO Integration Test")
    print("=" * 60)

    # Server URL (adjust if needed)
    server_url = "http://localhost:8000"

    try:
        print(f"\n🔌 Connecting to {server_url}...")
        await sio.connect(
            server_url,
            auth={"token": "test-token-123"},
            socketio_path="/socket.io/",
            wait_timeout=5,
        )

        # Wait for some data
        print("\n⏳ Waiting for real-time updates (30 seconds)...")
        await asyncio.sleep(30)

        # Disconnect
        print("\n👋 Disconnecting...")
        await sio.disconnect()

        print("\n✅ Test completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("\n⚠️  Make sure the backend is running with: make run-backend")
    print("⚠️  Or run the full stack with: make run-all\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)
