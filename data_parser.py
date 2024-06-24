import pandas as pd
import numpy as np
from os import listdir
import os
import sys
from data_handler import DataHandler


class DataParser():

    def __init__(self, source):
        self.source = source


    def create_data(self):
        data_handler_instance = DataHandler(self.source)
        data_handler_instance.get_create_data()
        data_handler_instance.dump_to_csv()
        self.create_inventory()
        self.randomize_data()
        self.create_full_list()


    def create_inventory(self):
        files = listdir('csv/')
        inventory = pd.DataFrame()
        for f in files:
            items = pd.read_csv(f'csv/{f}')
            id, product = items.columns[0], items.columns[1]
            new_df = items[[id, product]]
            inventory = pd.concat([inventory, new_df]).drop_duplicates(subset=[id]).reset_index(drop=True)


            inventory.to_csv('csv/inventory.csv', index = None, header=True)
            id_s = items[[id]] 
            id_s.to_csv('csv/ids.csv', index = None, header=True)

    def create_full_list():
        cities = ['Sofia', 'Burgas', 'Plovdiv', 'Ruse']
        years = ['2020', '2021', '2022', '2023']
        
        for city in cities:
            inventory = pd.DataFrame()
            for year in years:
                file_path = f'splits_data/{city}_{year}.csv'
                if os.path.exists(file_path):
                    print(file_path)
                    items = pd.read_csv(file_path)
                    inventory = pd.concat([inventory, items]).reset_index(drop=True)
            final_df = inventory.sort_values(['Артикул (код)', 'Година', 'Месец'])
            sales_df = final_df[['Артикул (код)', 'Количество', 'Месец',  'Година']]
            sales_df.to_csv(f'combined_data/{city}.csv', index = None, header=True)


    def get_unique_items_number(self, data):
        return data['Артикул (код)'].nunique()


    def create_data_for_model(self, file_name, file_path):
        items = pd.read_csv(file_path)
        data_dict = {}

        for _, row in items.iterrows():
            product_id = int(row['Артикул (код)'])
            data_dict[product_id] = np.zeros(48, dtype=np.int32)

        for _, row in items.iterrows():
            product_id = int(row['Артикул (код)'])
            items_sold = int(row['Количество'])
            month = int(row['Месец'])
            year = int(row['Година'])
            array = data_dict[product_id]
            array_index = (year % 10) * 12 + month - 1
            array[array_index] = items_sold
            data_dict[product_id] = array

        data_frame = pd.DataFrame.from_dict(data_dict)
        data_frame.to_csv(f'end_data/result_{file_name}.csv')
        return data_frame


if __name__ == '__main__':
    DataParser(sys.argv[1]).create_data()
