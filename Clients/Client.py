import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from Data import Data
from Model import MLPModel


class Client:
    def __init__(self, 
                 client_id,
                 data=Data(),
                 model_generator=MLPModel(),
                 epochs_per_round=1, 
                 malicious=None):
        self.model_generator = model_generator
        self.data = data
        self.epochs_per_round = epochs_per_round
        self.malicious = malicious
        self.x, self.y = data.get_client_data(client_id)

    def evaluate(self, parameters):
        model = self.model_generator.get_model(parameters)
        metrics = model.evaluate(self.x, self.y, verbose=0)
        self.model_generator.clear_session()
        return metrics

    def _get_gradients(self, old_model, new_model):
        if self.malicious is not None and 'scale_gradients' in self.malicious:
            scale = self.malicious['scale_gradients']
        else:
            scale = 1.0
        return [(old_layer - new_layer) * scale for old_layer, new_layer in zip(old_model, new_model)]

    def _generate_noise(self, parameters):
        noise = []
        for layer in parameters:
            noise.append(np.random.normal(scale=self.malicious['send_noise'], size=layer.shape).astype(np.float32))
        return noise

    def train(self, parameters):
        if self.malicious is not None and 'send_noise' in self.malicious:
            return self._generate_noise(parameters)
        model = self.model_generator.get_model(parameters)
        model.fit(self.x, self.y, epochs=self.epochs_per_round, verbose=0)
        weights = model.get_weights()
        self.model_generator.clear_session()
        return self._get_gradients(parameters, weights)


if __name__ == '__main__':
    data = Data()
    model = MLPModel()
    cliente = Client(client_id=1, model_generator=model, malicious={''})
