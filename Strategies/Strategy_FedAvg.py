from random import sample
from sklearn.metrics import confusion_matrix
import multiprocessing
import numpy as np
import sys


class FedAVG:
    def __init__(self, model_generator, clients, data, max_rounds, logger, **kwargs):
        self.model_generator = model_generator
        self.clients = clients
        self.data = data
        self.max_rounds = max_rounds
        self.logger = logger
        self.n_clients = kwargs.get('n_clients', 3)
        self.parameters = kwargs.get('initial_parameters', self.model_generator.get_parameters())
        self._round = 0

    def client_selection(self):
        selected = sample(self.clients.clients, self.n_clients)
        self.logger(f"Selected clients: {[client.client_id for client in selected]}")
        return selected

    @staticmethod
    def _get_bites(model):
        bites = 0
        if isinstance(model, list):
            for layer in model:
                bites += sys.getsizeof(layer)
        else:
            bites += sys.getsizeof(model)
        return bites

    def train_on_clients(self, clients):
        received_params = []
        params = self.parameters

        self.logger("Starting training")
        self.logger.transmission("SERVER", "CLIENT", "Global params", self._round + 1, self._get_bites(params)*len(clients))
        #for client in clients:
        #    received_params.append(
        #        client.train(params)
        #    )
        received_params = self._train_on_clients(clients, params)
        self.logger.transmission("CLIENT", "SERVER", "Trained models", self._round + 1, self._get_bites(received_params[-1]) * len(clients))
        self.logger("Training Finished")
        return received_params

    def model_aggregation(self, received_params):
        self.logger("Aggregating parameters")
        for layer in range(len(received_params[0])):
            self.parameters[layer] -= np.average([client_model[layer] for client_model in received_params], axis=0)
        self.logger("Aggregation Finished")

    def run(self):
        for self._round in range(self.max_rounds):
            self.logger(f"Starting round {self._round+1}")
            clients = self.client_selection()
            received_params = self.train_on_clients(clients)
            self.model_aggregation(received_params)

            loss, accuracy = self.evaluate()
            self.logger.metrics(self._round+1, loss, accuracy)
        self.logger("Federated Training Finished")

    def evaluate(self):
        x, y = self.data.get_server_data()
        model = self.model_generator.get_model(self.parameters)
        metrics = model.evaluate(x, y, verbose=0)
        return metrics

    def confusing_matrix(self):
        x, y = self.data.get_server_data()
        model = self.model_generator.get_model(self.parameters)
        y_pred = np.argmax(model.predict(x, verbose=0), axis=1)
        confusion = confusion_matrix(y, y_pred)
        return confusion

    @staticmethod
    def _train_on_clients(clients, params):
        def wrapper(client, params, new_params):
            new_params[client.client_id] = client.train(params)

        manager = multiprocessing.Manager()
        new_params = manager.dict()
        p_list = [multiprocessing.Process(target=wrapper, args=(client, params, new_params)) for client in clients]
        [p.start() for p in p_list]
        [p.join() for p in p_list]
        return new_params.values()
