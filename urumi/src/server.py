from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import asyncio
import os

app = FastAPI()

# We will set this from main.py
TELEMETRY_QUEUE = None

def set_telemetry_queue(q):
    global TELEMETRY_QUEUE
    TELEMETRY_QUEUE = q

# Ensure static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get():
    with open("static/index.html", "r") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            if TELEMETRY_QUEUE and not TELEMETRY_QUEUE.empty():
                data = TELEMETRY_QUEUE.get_nowait()
                await websocket.send_json(data)
            else:
                await asyncio.sleep(0.05) # 20Hz polling
    except WebSocketDisconnect:
        print("Client disconnected")
