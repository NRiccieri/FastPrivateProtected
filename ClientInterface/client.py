import requests
import numpy as np


class Client:
    def __init__(self, client_id, address, port, malicious):
        self.client_id = client_id
        self.address = address
        self.port = port
        self.malicious = malicious

    def evaluate(self, parameters):
        metrics = self.__communicate(
            'evaluate',
            parameters
        )
        if metrics is None:
            return None
        return metrics

    def train(self, parameters):
        new_parameters = self.__communicate(
            'train',
            parameters
        )
        if new_parameters is None:
            return None
        return [np.array(layer, dtype=np.float32) for layer in new_parameters]

    def __communicate(self, endpoint, parameters):
        answer = requests.post(
            f'http://{self.address}:{self.port}/{endpoint}',
            json={
                'parameters': [np.fmax(np.fmin(layer, 1e15), -1e15).tolist() for layer in parameters]
            }
        )
        if answer.status_code != 200:
            return None
        return answer.json()
