from sarima_model import SarimaModel
from rnn_method import RNNModel
from prophet_model import Prophet
from linear_method import LinearRegression
import sys


class DataPredictor():
    def __init__(self, model_choice):
        self.model_choice = model_choice
        self.run_model()

    def run_model(self):
        if self.model_choice == 'SARIMA':
            SarimaModel()
        elif self.model_choice == 'RNN':
            RNNModel()
        elif self.model_choice == 'Prophet':
            Prophet()
        elif self.model_choice == 'LinearRegression':
            LinearRegression()


if __name__ == '__main__':
    DataPredictor(sys.argv[1]).create_data()
