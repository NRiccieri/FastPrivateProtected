import datetime
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import seaborn as sns


class Logger:
    def __init__(self, experiment_name, parameters, print_log=True):
        self.print_log = print_log
        self.experiment_name = experiment_name
        self.strategy = parameters["strategy"]
        self.output_file = f'./Experiments_logs/{experiment_name}_{self.strategy}_{self._get_time(format="%Y%m%d_%H%M%S")}.log'
        self.json_file = self.output_file[:-3]+'json'
        self.confusion_file = self.output_file[:-3]+'png'
        self.json = {'experiment_name': experiment_name,
                     'parameters': parameters,
                     'participants': {
                         'clients': [],
                         'malicious': []
                     },
                     'rounds': defaultdict(lambda: {'server_client': 0, 'client_server': 0})}
        self._log(f'[EXPERIMENT] {self.experiment_name}')
        self._log(f'[PARAMETERS] {parameters}')

    def __call__(self, msg):
        self._log(msg)

    def _log(self, message):
        if self.print_log:
            print(f'[{self._get_time()}][{self.experiment_name}][{self.strategy}] {message}')
        with open(self.output_file, 'a') as output:
            output.write(f'[{self._get_time()}] {message}\n')

    def log_clients(self, clients):
        for client in clients:
            if client.malicious:
                self.json['participants']['malicious'].append(client.client_id)
            else:
                self.json['participants']['clients'].append(client.client_id)
        self._log(f"[CLIENTS]: {self.json['participants']['clients']}")
        self._log(f"[ATTACKERS]: {self.json['participants']['malicious']}")

    def metrics(self, round_, loss, accuracy):
        self.json['rounds'][round_].update({'loss': loss, 'accuracy': accuracy})
        self._log(f"[METRICS][ROUND {round_}] LOSS: {round(loss, 3)}   ACCURACY: {round(accuracy, 3)}")

    def transmission(self, source, destination, transmission_type, round_, bits):
        self.json['rounds'][round_][f'{source.lower()}_{destination.lower()}'] += bits
        self._log(f"[TRANSMISSION][ROUND {round_}] [{source} -> {destination}] {bits} ({transmission_type})")

    def reputations(self, reputation_list, round_):
        reputation_list = [round(r, 4) for r in reputation_list]
        self.json['rounds'][round_]['reputation'] = reputation_list
        self._log(f"[REPUTATIONS][ROUND {round_}] {reputation_list}")

    def save_json(self):
        with open(self.json_file, 'w') as json_output:
            json_output.write(json.dumps(self.json, indent=4))

    def model_evaluation(self, c, round_):
        self.json['rounds'][round_]['model_eval'] = c

    @staticmethod
    def _get_time(format="%d/%m/%Y %H:%M:%S"):
        now = datetime.datetime.now()
        return now.strftime(format)
