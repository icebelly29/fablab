import multiprocessing as mp
import asyncio
import logging
from queue import Empty
import uvicorn

from server import app, set_telemetry_queue
from camera import capture_loop
from grbl import GrblController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MAIN")

def vision_worker(cmd_queue, telemetry_queue):
    """
    Runs in a separate process to avoid GIL bottlenecks.
    Handles camera capture, image processing, and ML.
    """
    logger.info("Starting Vision Worker Process...")
    # This loop blocks the process, constantly reading camera and processing.
    capture_loop(cmd_queue, telemetry_queue)

async def main():
    logger.info("Initializing Main Controller...")
    
    # Queues for IPC
    cmd_q = mp.Queue()
    telem_q = mp.Queue(maxsize=2) # Keep maxsize small for lowest latency
    
    # Start Vision Process
    vision_proc = mp.Process(target=vision_worker, args=(cmd_q, telem_q), daemon=True)
    vision_proc.start()
    
    # Setup GRBL Controller
    grbl = GrblController(port='COM3', baudrate=115200) # Adjust for Pi, e.g., /dev/ttyUSB0
    # await grbl.connect()
    
    # Give the FastAPI server access to the telemetry queue
    set_telemetry_queue(telem_q)
    
    # Run FastAPI server
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    
    # Concurrently run server and GRBL monitoring
    # await asyncio.gather(server.serve(), grbl.run_loop())
    await server.serve()
    
    # Cleanup
    vision_proc.terminate()

if __name__ == '__main__':
    # Fix for multiprocessing on Windows
    mp.freeze_support()
    asyncio.run(main())
