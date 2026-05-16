import sys
sys.path.append('.')
from server.uart_comm import UARTComm
uart = UARTComm(mock=True)
uart.configure(3, 2, 160, 160)
uart._pos_x_steps = -23552
uart.send_move_to(-24704, 600)
