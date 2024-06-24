import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')


class SarimaModel():

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


    def sarima_forecast(ts, steps=12):
        model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        result = model.fit(disp=False)
        forecast = result.get_forecast(steps=steps)
        return forecast.predicted_mean


    def create_forecast(self):
        forecast_df = pd.DataFrame()
        for col in self.df.columns:
            forecast_df[col] = self.sarima_forecast(self.df[col])
        forecast_df.index = pd.date_range(start=self.df.index[-1] + pd.offsets.MonthEnd(1), periods=12, freq='M')
        forecast_df = forecast_df.round().astype(int)
        forecast_df.to_csv('predictions/sarima_forecast_next_year.csv', index=True)


