import numpy as np
from . import Strategy_FedAvg


class Median(Strategy_FedAvg.FedAVG):
    def model_aggregation(self, received_params):
        self.logger("Aggregating parameters")
        for layer in range(len(received_params[0])):
            self.parameters[layer] -= np.median([client_model[layer] for client_model in received_params], axis=0)
        self.logger("Aggregation Finished")
