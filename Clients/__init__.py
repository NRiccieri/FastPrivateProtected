import os
from flask import Flask, request
import json
import argparse
from Client import Client
import requests
import time
import numpy as np


app = Flask(__name__)
client = None


@app.route('/ping', methods=['GET'])
def ping():
    return 'PING'


@app.route('/info', methods=['GET'])
def info():
    return {
        'server_url': args['server_url'],
        'client_id': args['client_id'],
        'client_port': args['client_port'],
        'malicious': malicious
    }


@app.route('/register', methods=['GET'])
def register():
    register(
        args['server_url'],
        args['client_id'],
        args['client_port'],
        bool(malicious)
    )
    return "SUCCESS"


@app.route('/evaluate', methods=['POST'])
def evaluate():
    parameters = request.json['parameters']
    parameters = [np.array(layer, dtype=np.float32) for layer in parameters]
    return client.evaluate(parameters)


@app.route('/train', methods=['POST'])
def train():
    parameters = request.json['parameters']
    parameters = [np.array(layer, dtype=np.float32) for layer in parameters]
    new_parameters = client.train(parameters)
    new_parameters = [layer.tolist() for layer in new_parameters]
    return json.dumps(new_parameters)


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--client-port', '-p', type=int)
    parser.add_argument('--server-url', '-s', type=str)
    parser.add_argument('--client_id', '-i', type=int)
    parser.add_argument('--send_noise', type=float, default=None)
    parser.add_argument('--scale_gradients', type=float, default=None)
    args = parser.parse_args()

    args_dict = {
        'client_port': _get_env('CLIENT_PORT', args.client_port, int),
        'server_url': _get_env('SERVER_URL', args.server_url),
        'client_id': _get_env('CLIENT_ID', args.client_id, int),
        'malicious': {
            'send_noise': _get_env('SEND_NOISE', args.send_noise, float),
            'scale_gradients': _get_env('SCALE_GRADIENTS', args.scale_gradients, float)
        }
    }

    return args_dict


def register(server_url, client_id, client_port, malicious):
    while True:
        try:
            answer = requests.get(
                f'http://{server_url}',
                json={
                    'client_id': client_id,
                    'client_port': client_port,
                    'malicious': malicious
                }
            )
        except Exception:
            time.sleep(3)
        else:
            if answer.status_code != 200:
                time.sleep(3)
            else:
                break


def _get_env(name, default=None, convert=None):
    value = os.getenv(name, default)
    if value is not None and convert:
        value = convert(value)
    return value


if __name__ == '__main__':
    args = process_args()

    if not args['client_port']:
        raise Exception("Port not defined")
    if not args['server_url']:
        raise Exception("Server not defined")
    if args['client_id'] is None:
        raise Exception("Client ID not defined")
    
    malicious = {k: v for k, v in args['malicious'].items() if v is not None}
    
    client = Client(
        client_id=args['client_id'],  
        malicious=malicious
    )

    app.run(host='0.0.0.0', port=args['client_port'], debug=True)
