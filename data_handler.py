import pandas as pd
import numpy as np
from os import listdir


class DataHandler():

    def __init__(self, source):
        self.source = source
        self.data = None


    def create_data(self):
        if self.source in 'excel':
            self.create_csv()
        else:
            self.get_db_data()
            self.dump_csv()
    

    def dump_csv(self):
       data = pd.DataFrame(self.data)
       data.to_csv(f'csv/data.csv', index = None, header=True)

    
    def create_csv(self):
        files = listdir('excel/')
        
        for f in files:
            print(f'Parsing {f}:')
            xlsx_data = pd.read_excel(f'excel/{f}')
            print(f'Saving {f} in csv/{f}:')
            xlsx_data.to_csv(f'csv/{f.strip('.xlsx')}.csv', index = None, header=True)




    def get_db_data(self):
        pass

