import socket
import requests
import time
import json
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from .client import Client

CLIENTS_FILE_PATH = os.path.join(os.path.dirname(__file__), '__clients.json')


class ClientSet:
    def __init__(self, minimum, desired, port, timeout, logger):
        self.clients = []
        self.logger = logger
        self.load_file()
        if len(self.clients) < desired:
            self.registry(minimum, desired, port, timeout)
            self.update_file()
        elif len(self.clients) > desired:
            self.clients = self.clients[:desired]

    def load_file(self):
        try:
            with open(CLIENTS_FILE_PATH, 'r') as clients_file:
                clients_data = json.load(clients_file)
        except Exception:
            return None
        for client_data in clients_data['clients']:
            if self.__ping(client_data):
                self.clients.append(
                    Client(
                        client_id=client_data['client_id'],
                        address=client_data['address'],
                        port=client_data['client_port'],
                        malicious=client_data['malicious'],
                    )
                )
                self.logger(f"Connected to {client_data['client_id']} at " 
                            f"{client_data['address']}:{client_data['client_port']}")

    def update_file(self):
        with open(CLIENTS_FILE_PATH, 'w') as clients_file:
            clients_file.write(
                json.dumps({'clients': [
                    {'client_id': client.client_id,
                     'address': client.address,
                     'client_port': client.port,
                     'malicious': client.malicious}
                    for client in self.clients]}
                )
            )

    def registry(self, minimum, desired, port, timeout):
        start_time = time.time()

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('localhost', port))

        server_socket.listen()

        self.logger(f"Listening on port {port}...")

        try:
            while len(self.clients) < desired:
                client_socket, address = server_socket.accept()
                data = client_socket.recv(1024).decode('utf-8')

                if data:
                    json_data = json.loads(data.split('\r\n\r\n')[-1])
                    self.clients.append(
                        Client(
                            client_id=json_data['client_id'],
                            address=address[0],
                            port=json_data['client_port'],
                            malicious=json_data['malicious'],
                        )
                    )
                    self.logger(f"Connected to {json_data['client_id']} at {address[0]}:{json_data['client_port']}")
                    client_socket.send(b'HTTP/1.1 200 OK\r\n\r\n')
                else:
                    continue
                if time.time() - start_time >= timeout and len(self.clients) >= minimum:
                    self.logger("Timeout reached. Stopping...")
                    break
        finally:
            time.sleep(10)
            server_socket.close()
            self.logger("Server closed.")

    def __len__(self):
        return len(self.clients)

    def __getitem__(self, index):
        return self.clients[index]

    @staticmethod
    def __ping(client_data):
        try:
            answer = requests.get(
                f'http://{client_data["address"]}:{client_data["client_port"]}/ping',
            )
            return answer.status_code == 200
        except Exception:
            return False
