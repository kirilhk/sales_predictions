import pandas as pd
import numpy as np
from prophet import Prophet


class Prophet():

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


    def prophet_forecast(ts, steps=12):
        ts_df = ts.reset_index()
        ts_df.columns = ['ds', 'y']
        
        model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
        model.fit(ts_df)

        future = model.make_future_dataframe(periods=steps, freq='M')
        forecast = model.predict(future)
        return forecast[['ds', 'yhat']].tail(steps)


    def create_forecast(self):
        forecast_df = pd.DataFrame()
        for col in self.df.columns:
            forecast = self.prophet_forecast(self.df[col])
            forecast_df[col] = forecast['yhat'].values

        forecast_df.index = pd.date_range(start=self.df.index[-1] + pd.offsets.MonthEnd(1), periods=12, freq='M')
        forecast_df = forecast_df.round().astype(int)
        forecast_df.to_csv('predictions/prophet_forecast_next_year.csv', index=True)





