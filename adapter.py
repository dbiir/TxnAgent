#!/usr/bin/python3
import argparse
import os
import signal
import socket

from agent.agent import TxnAgent

server_sockets: list[socket.socket] = []
client_sockets: list[socket.socket] = []
workloads = ["ycsb", "tpcc", "smallbank"]

def prepare_for_connect():
    # Create a socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the address and port
    server_address = ('localhost', 7654)
    server_socket.bind(server_address)

    # Listen for connections
    server_socket.listen(1)
    print('Waiting for connection...', flush=True)

    return server_socket


def signal_handler(signal, frame):
    close_service(server_sockets, client_sockets)


def close_service(server_ss: list[socket.socket], client_ss: list[socket.socket]):
    for s in server_ss:
        s.close()
    for s in client_ss:
        s.close()


# register signal handler
signal.signal(signal.SIGINT, signal_handler)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--workload", dest='wl', choices=workloads, type=str, required=True,
                        help="specify the workload")
    parser.add_argument("-f", "--filepath", dest='fp', type=str, required=False,
                        help="file path for offline training data")
    parser.add_argument("-p", "--phase", dest='phase', choices=['offline', 'online'], type=str, required=False,
                        help="specify the phase: offline or online")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    txn_service = TxnAgent() 

    if args.phase == "offline":
        txn_service.offline_train()
    server_socket = prepare_for_connect()
    server_sockets.append(server_socket)

    # Accept a connection
    client_socket, client_address = server_socket.accept()
    print('Connection established:', client_address, flush=True)
    client_sockets.append(client_socket)

    # Receive and send messages
    while True:
        data = client_socket.recv(10240).decode()
        if not data:
            break
        print('Received message:', data, flush=True)
        variables: list[str] = data.split(",")
        if variables[0] == "close":
            print("Received close command, shutting down...", flush=True)
            txn_service.export_metrics()
            txn_service.writer.close()
            # Save final checkpoint
            ckpt_dir = os.path.join(os.path.dirname(__file__), 'models')
            os.makedirs(ckpt_dir, exist_ok=True)
            txn_service.rl_agent.save(os.path.join(ckpt_dir, 'final_online.pt'))
            print("Model saved and TensorBoard closed.", flush=True)
            break
        elif variables[0] == "online":
            filename = variables[1]
            response: str = txn_service.service(filename, args.wl)
            client_socket.sendall(response.encode("utf-8"))
        else:
            print("Unknown command:", variables[0], flush=True)
            client_socket.sendall("error: unknown command".encode("utf-8"))

    # Close the connection
    close_service(server_sockets, client_sockets)
