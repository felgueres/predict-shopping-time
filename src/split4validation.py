'''
This file serves to split the training dataset into in training and validation set.
'''

import pandas as pd
from os import path, listdir

def split(datapath, split_frac = 0.92, validation = True):
    '''
    Splits data in validation / traning.

    Parameters
    ----------
    datapath: string
        Path to training data file.

    Returns
    -------
    data_train_order
    data_train_trips
    data_val_
    data_val_trips: dataframes
    '''
    #Get file names
    files = [path.join(datapath, file) for file in sorted(listdir(datapath)) if file.endswith('csv')]
    #Load data to dataframes
    df_order_items = pd.read_csv(files[0])
    df_train_trips = pd.read_csv(files[2])
    #Split the files by trip ids
    trip_ids_train = df_train_trips.trip_id.sample(frac = split_frac, random_state = 10)
    trip_ids_val   = df_train_trips[~df_train_trips.isin(trip_ids_train)].trip_id
    #Split datasets
    train_order = df_order_items.loc[df_order_items.trip_id.isin(trip_ids_train)]
    train_trips = df_train_trips.loc[df_train_trips.trip_id.isin(trip_ids_train)]
    val_order = df_order_items.loc[df_order_items.trip_id.isin(trip_ids_val)]
    val_trips = df_train_trips.loc[df_train_trips.trip_id.isin(trip_ids_val)]

    val_y_test = pd.DatetimeIndex(val_trips.shopping_ended_at) - pd.DatetimeIndex(val_trips.shopping_started_at)
    #Convert to seconds
    val_y_test = val_y_test.seconds

    val_trips.pop('shopping_ended_at')
    
    return train_order, train_trips, val_order, val_trips, val_y_test

if __name__ == '__main__':
    pass
