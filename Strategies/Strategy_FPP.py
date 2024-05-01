import random
import numpy as np
import multiprocessing

from . import Strategy_FedAvg


class FPP(Strategy_FedAvg.FedAVG):
    def __init__(self, model_generator, clients, data, max_rounds, logger, **kwargs):
        self.initial_selection = kwargs.get('initial_selection', 15)
        self.reputation_recover = kwargs.get('reputation_recover', 1.2)
        self.reputation_penalty = kwargs.get('reputation_penalty', 0.98)
        self.recover_threshold = kwargs.get('recover_threshold', 1.1)
        for client in clients:
            client.reputation = 1.0
            client.suspicion_factor = 0
        self.previous_parameters = None
        self.previous_selected = None
        self.previous_loss = None
        super().__init__(model_generator, clients, data, max_rounds, logger, **kwargs)

    def client_selection(self):
        self.logger.reputations([c.reputation for c in self.clients.clients], self._round + 1)
        initial_selected = self._sample(self.clients.clients, self.initial_selection)
        self.logger.transmission("SERVER", "CLIENT", "Prior evaluation", self._round + 1, self._get_bites(self.parameters) * len(initial_selected))
        evaluations = self._evaluate_clients(initial_selected, self.parameters)
        loss_list = [e[1] for e in evaluations]

        if self._round == 0:  # Save the base model in the first round
            self._save_checkpoint(self._estimate_loss(loss_list))
        elif self.previous_selected is not None:  # Evaluate if model was trained in the last selection.
            recovered = self._evaluate_model(loss_list)
            if recovered:
                return self.client_selection()
            else:
                self._save_checkpoint(self._estimate_loss(loss_list))
        evaluations.sort(key=lambda w: w[1], reverse=True)
        self.logger(f"Initial selection: {[(client.client_id, evaluation) for client, evaluation in evaluations]}")
        selected = [client[0] for client in evaluations[:self.n_clients]]
        self.previous_selected = [c.client_id for c in selected]
        self.logger(f"Selected clients: {[client.client_id for client in selected]}")
        return selected

    @staticmethod
    def _sample(candidates, k):
        candidates = candidates[::]
        p = [candidate.reputation for candidate in candidates]
        selected = []
        while len(selected) < k:
            i = random.choices(range(len(candidates)), weights=np.array(p) / sum(p))[0]
            selected.append(candidates[i])
            candidates.pop(i)
            p.pop(i)
        return selected

    @staticmethod
    def _copy_variable(variable):
        new_variable = []
        for layer in variable:
            new_variable.append(layer.copy())
        return new_variable

    def _evaluate_model(self, loss_list):
        estimated_loss = self._estimate_loss(loss_list)
        self.logger(f"Current estimated loss: {estimated_loss}")
        if estimated_loss > self.previous_loss * self.recover_threshold:  # Problem detected
            self.logger(f"The loss ABOVE recover threshold {round(estimated_loss, 3)}>{round(self.previous_loss * self.recover_threshold, 3)} ({round(estimated_loss / self.previous_loss, 3)}). RECOVERING")
            self.logger.model_evaluation('severe', self._round)
            self._recover_checkpoint()
            for client in self.clients.clients:
                if client.client_id in self.previous_selected:
                    client.suspicion_factor += 1
                    client.reputation *= self.reputation_penalty
            self.previous_selected = None
            return True
        # No problem detected, recover reputation
        else:
            self.logger(f"[LOSS ESTIMATION] The loss bellow penalty threshold {round(estimated_loss, 3)}<={round(self.previous_loss * self.recover_threshold, 3)} ({round(estimated_loss / self.previous_loss, 3)}).")
            self.logger.model_evaluation('approved', self._round)
            for client in self.clients.clients:
                if client.client_id in self.previous_selected:
                    client.suspicion_factor = 0
                    client.reputation = min(1.0, client.reputation * self.reputation_recover)
            return False

    def _save_checkpoint(self, estimated_loss):
        # Copy parameters and update loss
        self.previous_parameters = self._copy_variable(self.parameters)
        self.previous_loss = estimated_loss

    def _recover_checkpoint(self):
        self.parameters = self._copy_variable(self.previous_parameters)

    @staticmethod
    def _estimate_loss(loss_list):
        return np.average(sorted(loss_list))

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
