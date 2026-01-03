
import asyncio
import websockets
import json

async def handler(websocket):
    """
    Handles incoming WebSocket connections and messages.
    """
    print(f"Client connected: {websocket.remote_address}")
    try:
        # This will loop until the client disconnects
        async for message in websocket:
            try:
                data = json.loads(message)
                print(f"Received message: {data}")
            except json.JSONDecodeError:
                print(f"Received non-JSON message: {message}")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Client disconnected: {e.reason} (Code: {e.code})")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print(f"Connection closed for {websocket.remote_address}")

async def main():
    """
    Starts the WebSocket server.
    """
    host = "localhost"
    port = 8080
    print(f"Starting WebSocket server on ws://{host}:{port}...")
    # The `websockets.serve` function creates and starts the server
    async with websockets.serve(handler, host, port):
        await asyncio.Future()  # This keeps the server running indefinitely

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer is shutting down.")
