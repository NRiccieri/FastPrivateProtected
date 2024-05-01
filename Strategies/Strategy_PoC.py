from . import Strategy_FedAvg
from random import sample
import multiprocessing


class PowerOfChoice(Strategy_FedAvg.FedAVG):
    def __init__(self, model_generator, clients, data, max_rounds, logger, **kwargs):
        self.initial_selection = kwargs.get('initial_selection', 10)
        super().__init__(model_generator, clients, data, max_rounds, logger, **kwargs)

    def client_selection(self):
        initial_selected = sample(self.clients.clients, self.initial_selection)
        self.logger.transmission("SERVER", "CLIENT", "Prior evaluation",
                                 self._round + 1, self._get_bites(self.parameters) * len(initial_selected))
        #evaluations = [(client, client.evaluate(self.parameters)[0]) for client in initial_selected]
        evaluations = self._evaluate_clients(initial_selected, self.parameters)
        evaluations.sort(key=lambda w: w[1], reverse=True)
        self.logger(f"Initial selection: {[(client.client_id, evaluation) for client, evaluation in evaluations]}")
        selected = [client[0] for client in evaluations[:self.n_clients]]
        self.logger(f"Selected clients: {[client.client_id for client in selected]}")
        return selected

    @staticmethod
    def _evaluate_clients(clients, params):
        def wrapper(client, params, evaluations):
            evaluations[client.client_id] = (client, client.evaluate(params)[0])

        manager = multiprocessing.Manager()
        evaluations = manager.dict()
        p_list = [multiprocessing.Process(target=wrapper, args=(client, params, evaluations)) for client in clients]
        [p.start() for p in p_list]
        [p.join() for p in p_list]
        return evaluations.values()

