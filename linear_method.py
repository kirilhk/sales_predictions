import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class LinearRegression():

    def __init__(self, data_file):
        self.data_file = data_file
        self.df = None
        self.run()

    def run(self):
        self.load_data()
    

    def load_data(self):
        self.df = pd.read_csv('result_Burgas.csv')
        self.df.index = pd.date_range(start='2020-01-01', periods=48, freq='M')
        self.df['time'] = np.arange(len(self.df))


    def linear_regression_forecast(self, ts, steps=12):
        time = ts.index.values.reshape(-1, 1)
        sales = ts.values

        scaler = StandardScaler()
        time_scaled = scaler.fit_transform(time)

        model = LinearRegression()
        model.fit(time_scaled, sales)

        future_time = np.arange(len(ts), len(ts) + steps).reshape(-1, 1)
        future_time_scaled = scaler.transform(future_time)
        forecast = model.predict(future_time_scaled)
        return forecast


    def create_forecast(self):
        forecast_df = pd.DataFrame()
        for col in self.df.columns[:-1]:
            forecast_df[col] = self.linear_regression_forecast(self.df[col])
        forecast_df.index = pd.date_range(start=self.df.index[-1] + pd.offsets.MonthEnd(1), periods=12, freq='M')
        forecast_df = forecast_df.round().astype(int)
        forecast_df.to_csv('predictions/linear_regression_forecast_next_year.csv', index=True)
