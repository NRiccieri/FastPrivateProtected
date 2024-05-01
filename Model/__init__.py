import tensorflow as tf
import datetime
from tensorflow.keras.models import Sequential
import os


PATH = os.path.dirname(os.path.abspath(__file__))


class MLPModel:
    def __init__(self, training_parameters=None):
        if training_parameters is None:
            self.training_parameters = {"learning_rate": 1e-3}
        else:
            self.training_parameters = training_parameters
        self.model = Sequential([
            tf.keras.layers.Dense(265, input_shape=(10,)),  # Input Layer
            tf.keras.layers.Dense(265, activation='relu'),  # First Hidden Layer
            tf.keras.layers.Dense(265, activation='relu'),  # Second Hidden Layer
            tf.keras.layers.Dense(6, activation='softmax')  # Output
        ])

        self.compile()

    def get_parameters(self):
        return self.model.get_weights()

    def get_model(self, parameters=None, compile=True):
        if parameters:
            self.model.set_weights(parameters)
            if compile:
                self.compile()
        return self.model

    def clear_session(self):
        tf.keras.backend.clear_session()

    def compile(self):
        self.model.compile(optimizer=tf.keras.optimizers.SGD(**self.training_parameters),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def create_initial_model(self):
        self.model.save(f'{PATH}/Models/experiments_initial')

    def get_initial_model(self):
        self.model = tf.keras.models.load_model(f'{PATH}/Models/experiments_initial')
        return self.model.get_weights()

    @staticmethod
    def get_id():
        if not os.path.exists(f'{PATH}/Models/experiments_initial'):
            return None
        timestamp = os.path.getmtime(f'{PATH}/Models/experiments_initial')
        return datetime.datetime.fromtimestamp(timestamp).strftime("%Y%m%d%H%M")


if __name__ == '__main__':
    model = MLPModel().get_model()
    model.summary()
