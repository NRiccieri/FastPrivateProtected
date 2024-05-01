import pandas as pd
import numpy as np
import datetime
import json

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from scipy.stats import ks_2samp
import seaborn as sns
import os

PATH = os.path.dirname(os.path.abspath(__file__))


class Data:
    data_file = f'{PATH}/files/data.json'

    def __init__(self, n_clients=None):
        self.x, self.y, self.subject = self._read_file(self.data_file)

        self._encoder = preprocessing.LabelEncoder()
        self.y = self._encoder.fit_transform(self.y)

        self._scaler = StandardScaler()
        self.x = self._scaler.fit_transform(self.x)

        self.n_clients = n_clients
        self._reset_subjects()

    def get_client_data(self, client_id):
        return self.x[self.subject == client_id], self.y[self.subject == client_id]

    def get_server_data(self):
        return self.x, self.y

    @staticmethod
    def _read_file(file):
        with open(file, 'rb') as input_file:
            data = json.loads(input_file.read())

        data_table = []
        for client in data['users']:
            for x, y in zip(data['user_data'][client]['x'],
                            data['user_data'][client]['y']):
                data_table.append([client] + x + [y])
        df = pd.DataFrame(data_table, columns=['subject'] + [f'x_{i}' for i in range(len(x))] + ['y'])

        x = pd.DataFrame(df.drop(['subject', 'y'], axis=1))
        subject = df.subject
        y = df.y.values.astype(object)
        return x, y, subject

    def _reset_subjects(self):
        clients_list = self.subject.unique()
        self.n_clients = len(clients_list)
        self._clients_map = {i: j for i, j in zip(clients_list, range(len(clients_list)))}
        self.subject = self.subject.replace(self._clients_map)

    @staticmethod
    def get_id():
        if not os.path.exists(Data.train_file):
            return None
        timestamp = os.path.getmtime(Data.train_file)
        return "S" + datetime.datetime.fromtimestamp(timestamp).strftime("%Y%m%d%H%M")


if __name__ == '__main__':
    data = Data()
    ks_matrix = np.zeros((data.n_clients, data.n_clients))

    for i in range(data.n_clients):
        _, i_data_y = data.get_client_data(i)
        for j in range(i, data.n_clients):
            _, j_data_y = data.get_client_data(j)
            p_value_ij = ks_2samp(i_data_y, j_data_y, method='exact')
            ks_matrix[i][j] = p_value_ij[1]
            ks_matrix[j][i] = p_value_ij[1]

    # Create a dataset
    df = pd.DataFrame(ks_matrix)

    # Default heatmap
    p1 = sns.heatmap(df, cmap='viridis')
    p1.get_figure().savefig('data_heatmap.png')
