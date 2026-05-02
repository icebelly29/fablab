import asyncio
import logging
import serial_asyncio

logger = logging.getLogger("GRBL")

class GrblController:
    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.reader = None
        self.writer = None
        self.buffer_size = 128
        self.current_buffer_bytes = 0
        self.connected = False
        self.command_queue = asyncio.Queue()

    async def connect(self):
        try:
            self.reader, self.writer = await serial_asyncio.open_serial_connection(url=self.port, baudrate=self.baudrate)
            self.connected = True
            logger.info(f"Connected to GRBL on {self.port}")
            # Wake up GRBL
            self.writer.write(b"\r\n\r\n")
            await self.writer.drain()
            await asyncio.sleep(2)
            await self.clear_startup_buffer()
        except Exception as e:
            logger.error(f"Failed to connect to GRBL: {e}")

    async def clear_startup_buffer(self):
        while self.reader.in_waiting > 0:
            line = await self.reader.readline()
            logger.debug(f"GRBL Init: {line.strip()}")

    async def send_command(self, cmd: str):
        if not cmd.endswith('\n'):
            cmd += '\n'
        await self.command_queue.put(cmd)

    async def run_loop(self):
        if not self.connected:
            return
            
        while True:
            # Check for incoming responses
            if self.reader.in_waiting > 0:
                try:
                    line = await self.reader.readline()
                    line = line.decode('utf-8').strip()
                    if line == 'ok':
                        logger.debug("Received ok")
                    else:
                        logger.info(f"GRBL: {line}")
                except UnicodeDecodeError:
                    pass

            # Send commands if we have them
            if not self.command_queue.empty():
                cmd = await self.command_queue.get()
                self.writer.write(cmd.encode('utf-8'))
                await self.writer.drain()
                logger.debug(f"Sent: {cmd.strip()}")
                
            await asyncio.sleep(0.01)
