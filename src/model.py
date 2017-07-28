import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from preprocessing import Preprocess
from split4validation import split
from os import path, listdir
import time

start_time = time.time()

#Get file names
files = [path.join('../data/', file) for file in sorted(listdir('../data/')) if file.endswith('csv')]

#Read to dataframes
df_order_items = pd.read_csv(files[0])
df_test_trips = pd.read_csv(files[1])
df_train_trips = pd.read_csv(files[2])

#Separate for Preprocess consumption
train_order = df_order_items.loc[df_order_items.trip_id.isin(df_train_trips.trip_id)]
train_trips = df_train_trips.loc[df_train_trips.trip_id.isin(df_train_trips.trip_id)]

test_order = df_order_items.loc[df_order_items.trip_id.isin(df_test_trips.trip_id)]
test_trips = df_test_trips.loc[df_test_trips.trip_id.isin(df_test_trips.trip_id)]

#Init preprocessing class.
data = Preprocess(train_order, train_trips, test_order, test_trips)
#Fit
data.fit()
#Get data either scaled or not
X_train_scaled, X_test_scaled, test_ids, y_train_sqrt = data.get_data(scale = True)

#Training, using alpha from cross-validation
model = Lasso(alpha = 0.008377).fit(X_train_scaled, y_train_sqrt)

#Predict for X_test
y_pred_lasso = np.square(model.predict(X_test_scaled))

#Formating for challenge rules
test_ids = test_ids.to_frame()
test_ids['y_pred'] = y_pred_lasso
test_ids = test_ids.rename(columns={'y_pred': 'shopping_time'})
test_ids.set_index('trip_id', inplace=True)

#Export as csv
test_ids.to_csv('../predictions/predictions.csv')

print "-------%s seconds ------" %(time.time() - start_time)
