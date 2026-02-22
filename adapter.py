#!/usr/bin/python3
import argparse
import os
import signal
import socket
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from agent.agent import TxnAgent

server_sockets: list[socket.socket] = []
client_sockets: list[socket.socket] = []
txn_service: TxnAgent = None
workloads = ["ycsb", "tpcc", "smallbank"]

def prepare_for_connect():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_address = ('localhost', 7654)
    server_socket.bind(server_address)
    server_socket.listen(1)
    print('Waiting for connection...', flush=True)
    return server_socket


def graceful_shutdown(signum, frame):
    """Handle Ctrl+C: save model, export metrics, close sockets, then exit."""
    print(f"\n[adapter] Caught signal {signum}, shutting down...", flush=True)
    if txn_service is not None:
        try:
            txn_service.export_metrics()
            txn_service.writer.close()
            ckpt_dir = os.path.join(SCRIPT_DIR, 'models')
            os.makedirs(ckpt_dir, exist_ok=True)
            txn_service.rl_agent.save(os.path.join(ckpt_dir, 'final_online.pt'))
            print("[adapter] Model saved and TensorBoard closed.", flush=True)
        except Exception as e:
            print(f"[adapter] Error during cleanup: {e}", flush=True)
    for s in client_sockets:
        try:
            s.close()
        except Exception:
            pass
    for s in server_sockets:
        try:
            s.close()
        except Exception:
            pass
    print("[adapter] Sockets closed. Exiting.", flush=True)
    os._exit(0)


signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)


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
    txn_service = TxnAgent(workload=args.wl)

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
        data = client_socket.recv(10240).decode().strip()
        if not data:
            break
        print('Received message:', data, flush=True)
        variables: list[str] = data.split(",")
        if variables[0] == "close":
            print("Received close command, shutting down...", flush=True)
            txn_service.export_metrics()
            txn_service.writer.close()
            ckpt_dir = os.path.join(SCRIPT_DIR, 'models')
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

    # Normal close path
    for s in client_sockets:
        s.close()
    for s in server_sockets:
        s.close()
