import sys
sys.path.append('.')
from server.uart_comm import UARTComm

def test_jog():
    uart = UARTComm(mock=True)
    uart.configure(3, 2, 160, 160)
    uart.send_jog(1152, 0)
    print("Jog command built.")

def test_move():
    uart = UARTComm(mock=True)
    uart.configure(3, 2, 160, 160)
    uart._pos_x_steps = -25856
    uart._pos_y_steps = 600
    
    # We patch the mock so we can see what cmd is sent
    def mock_send(cmd):
        print("Move cmd:", cmd)
    uart._send_raw = mock_send
    
    # Bypass the mock=True block in send_move_to
    uart.mock = False
    uart.send_move_to(-24704, 600)

test_jog()
test_move()
