import numpy as np
from . import Strategy_FedAvg


class TrimmedMean(Strategy_FedAvg.FedAVG):
    def model_aggregation(self, received_params):
        self.logger("Aggregating parameters")
        for layer in range(len(received_params[0])):
            self.parameters[layer] -= np.average(
                np.sort([client_model[layer] for client_model in received_params], axis=0)[2:-2], axis=0)
        self.logger("Aggregation Finished")
