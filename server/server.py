import socket
import struct

import numpy as np

from controllers import *

clients = ['measure', 'control', 'weather']

N_horizon = 8
print(f"[*] Controller Horizon {N_horizon}")

HOST            = '127.0.0.1'
PORT = {
    'measure': 10000,
    'control': 10001,
    'weather': 10002
        }

print(f"[*] Server IP {HOST}")
print(f"[*] Measuring on port {PORT['measure']}")
print(f"[*] Controlling on port {PORT['control']}")
print(f"[*] Weather on port {PORT['weather']}")

sock = {}

# Create a socket for each of the clients and start listening
for client in clients:
    sock[client] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock[client].bind((HOST, PORT[client]))
    sock[client].listen()

temps_list = []

conn = {}

# Initialize controller

t_cols = []
w_cols = ['SolRad', 'OutsideTemp']
u_cols = ['SimulatedHeat']
y_cols = ['SimulatedTemp']

t_lags = 0
w_lags = 1
u_lags = 2
y_lags = 3

print(f"[*] t_lags: {t_lags}")
print(f"[*] w_lags: {w_lags}")
print(f"[*] u_lags: {u_lags}")
print(f"[*] y_lags: {y_lags}")

dict_cols = {
    't': (t_lags, t_cols),
    'w': (w_lags, w_cols),
    'u': (u_lags, u_cols),
    'y': (y_lags, y_cols)
}

controller = SVGP_MPCcontroller(
                dict_cols = dict_cols,
                N_horizon = N_horizon,
                recover_from_crash = False
)

# Enter TCP server loop
while True:
    # Connect to the clients
    for client in clients:
        print(f"[+] Waiting for a {client} connection...", end = ' ')
        iter_conn,_  = sock[client].accept()
        conn[client] = iter_conn
        print("Done")
    # Sending first control signal (initialization)
    print("[*] Entering the control loop")

    try:
        while True:

            ###
            # Begin timestep
            ###
            print("---")
            # Send control signal
            print("[*] Sending control signal...", end = ' ')
            u = controller.get_control_input()
            print(f"Applying control signal {u}")
            data = struct.pack(">d", u)
            conn['control'].sendall(data)
            print("Done")

            # Read the inputs and update controller measures


            # Read weather prediction
            weather = []
            print("[*] Reading weather measurement/prediction...", end = ' ')
            for idx in range((N_horizon + 1) * 2):
                weather_item = conn['weather'].recv(8)
                if weather_item:
                    weather_item = struct.unpack(">d", weather_item)
                    weather.append(weather_item)
                else:
                    break
            if len(weather) == ((N_horizon + 1) *2):
                weather = np.array(weather).reshape((2, N_horizon + 1)).T
            else:
                print("\nDid not get a complete weather prediction. Simulation ended?")
                break
            controller.add_disturbance_measurement(weather[0,:])
            controller.set_weather_forecast(weather[1:, :])
            print("Done")

            # Read temperature measurement
            print("Reading temperature measurement")
            data = conn['measure'].recv(8)
            if data:
                t_iter = struct.unpack(">d", data)[0]
                temps_list.append(t_iter)
            else:
                break
            print(f"Read temperature measurement {t_iter}")
            controller.add_output_measurement(t_iter)

            # Update the model since all data has been read
            controller.update_model()

    finally:
        print("[-] Closing connection to simulink")
        for client in clients:
            conn[client].close()
        print("[i] Dumping controller data")
        controller.save_data()
