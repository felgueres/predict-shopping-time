'''
Contains Preprocessing class to normalize and featurize data prior to modeling.
'''
from split4validation import split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class Preprocess(object):
    '''
    - Preprocessing of data prior to model fit
    - Returns feature engineered datasets
    - Handles dummy variables without leakages
    - Handles scaling

    Parameters
    ----------
    data_order_items: dataframe
        Contains items on each order

    data_trips_train / data_trips_test: dataframes
        Contains trips information

    '''

    def __init__(self, train_order_items, train_trips, test_order_items, test_trips):

        '''
        Initialize with string to data folder.
        '''
        self.train_order_items = train_order_items
        self.test_order_items = test_order_items
        self.df_train = train_trips
        self.df_test = test_trips
        self.y_train = None
        self.ids = None

    def _dtypes(self):
        '''
        Change date strings to DatetimeIndex.
        '''
        self.df_train.shopping_started_at = pd.DatetimeIndex(self.df_train.shopping_started_at)
        self.df_test.shopping_started_at = pd.DatetimeIndex(self.df_test.shopping_started_at)

        self.df_train.shopping_ended_at = pd.DatetimeIndex(self.df_train.shopping_ended_at)

    def _target(self):
        '''
        Compute target variable eg. shopping time when training.
        '''
        self.y_train = (self.df_train.shopping_ended_at - self.df_train.shopping_started_at).dt.seconds
        #Remove the shopping_ended_at feature from training dataset
        self.df_train.drop('shopping_ended_at', axis = 1, inplace = True)

    def _feature_basket_volume(self):
        '''
        Compute basket volume.
        '''
        def basket_qty(df, order_items):

            basket_volume = order_items.groupby('trip_id')['quantity'].sum().to_frame().reset_index().rename(columns = {'quantity': 'basket_qty'})
            #Merge to main
            df = df.merge(basket_volume, left_on = 'trip_id', right_on = 'trip_id', how = 'left')

            return df

        self.df_train = basket_qty(self.df_train, self.train_order_items)
        self.df_test = basket_qty(self.df_test, self.test_order_items)

    def _feature_distinct_items(self):
        '''
        Compute distinct items. by trip.
        '''

        def distinct_i(df, order_items):

            dist_item_cnt = order_items.groupby('trip_id')['item_id'].nunique().reset_index().rename(columns = {'item_id': 'dist_i_cnt'})
            df = df.merge(dist_item_cnt, left_on = 'trip_id', right_on = 'trip_id', how = 'left')

            return df

        self.df_train = distinct_i(self.df_train, self.train_order_items)
        self.df_test = distinct_i(self.df_test, self.test_order_items)

    def _feature_distinct_dpts(self):
        '''
        Compute distinct Dpts. by trip.
        '''

        def distinct_dpts(df, order_items):

            dpt_count = order_items.groupby('trip_id')['department_name'].nunique().reset_index().rename(columns ={'department_name': 'dist_dpts_cnt'})
            #MERGE TO MAIN
            df = df.merge(dpt_count, left_on = 'trip_id', right_on = 'trip_id', how = 'left')

            return df

        self.df_train = distinct_dpts(self.df_train, self.train_order_items)
        self.df_test = distinct_dpts(self.df_test, self.test_order_items)

    def _feature_reorder_factor(self):
        '''
        This is a shopper specific feature.

        The reorder factor given by It's the summation of reordered acounts normalized to the total number of products a person has ordered.
        Note a value of 1 is the floor for this feature.

        If on the test set there is a shopper that is not present on the training dataset, then a value of value is set.
        '''

        #Merge order information with train dataset
        reorders = self.train_order_items.merge(self.df_train.loc[:, ['trip_id', 'shopper_id']], left_on = 'trip_id', right_on = 'trip_id', how = 'left')
        #For each shopper, for each of item, compute the trips where it has been shopped.
        reordered_items = reorders.groupby(['shopper_id', 'item_id'])['trip_id'].apply(lambda x: x.count()).to_frame()
        #Calculate reordering factor.
        reordered_items = reordered_items.groupby('shopper_id')['trip_id'].sum() / reordered_items.groupby('shopper_id')['trip_id'].apply(lambda x: np.sum([np.ones((x.shape[0],1))]))
        reordered_factor = reordered_items.to_frame().rename(columns = {'trip_id': 'reorder_factor'})
        reordered_factor.reset_index(inplace =True)
        #At this point the dataframe has 2 columns: shopper_id and reorder_factor
        #No merge to train and and to test where applicable.

        self.df_train = self.df_train.merge(reordered_factor, left_on = 'shopper_id', right_on = 'shopper_id', how = 'left')
        self.df_test = self.df_test.merge(reordered_factor, left_on = 'shopper_id', right_on = 'shopper_id', how = 'left')
        #now fill in null on the test dataset, which would mean the user is not on the training dataset, as 1.
        self.df_test.reorder_factor.fillna(value = 1, inplace=True)

    def _feature_temporality(self):
        '''
        Time-based features.
        '''

        def temps(df):

            #Temp cols
            df['dow'] = df.shopping_started_at.dt.dayofweek
            df['hourofday'] = df.shopping_started_at.dt.hour

            #New features
            df['is_afternoon'] = np.where(df.hourofday < 13, 0, 1)
            df['busy_day'] = np.where(df.dow.isin([1,6]), 1, 0)
            df.drop(['dow','hourofday'], axis =1, inplace = True)

            return df

        self.df_train = temps(self.df_train)
        self.df_test = temps(self.df_test)

    def _store_id(self):
        '''
        Get dummies for departments.
        '''
        categories = np.union1d(self.df_train.store_id, self.df_test.store_id)
        train2dummify = self.df_train.store_id.astype('category', categories = categories)
        test2dummify = self.df_test.store_id.astype('category', categories = categories)

        train_dummies = pd.get_dummies(train2dummify, prefix = 'store')
        self.df_train = self.df_train.merge(train_dummies, left_index= True, right_index = True, how = 'left')
        self.df_train.drop('store_id', axis = 1, inplace = True)

        test_dummies = pd.get_dummies(test2dummify, prefix = 'store')
        self.df_test = self.df_test.merge(test_dummies, left_index= True, right_index = True, how = 'left')
        self.df_test.drop('store_id', axis = 1, inplace = True)

    def _featurize(self):
        '''
        Create features for model.
        '''
        self._feature_basket_volume()
        self._feature_distinct_items()
        self._feature_distinct_dpts()
        self._feature_reorder_factor()
        self._feature_temporality()
        self._store_id()

    def fit(self):
        '''
        Fit preprocessing methods
        '''
        self._dtypes()
        self._target()
        self._featurize()

    def get_data(self, scale = False):
        '''
        Returns a feature-engineered version of X and the transformed variable y.

        Usage for an already fitted preprocess class.

        X_train, X_test, test_ids, y_train = preprocess.get_data(scale=True)

        '''
        #Make copies so featurization only happens once.
        df_train = self.df_train.drop('trip_id', axis = 1).copy()
        df_test = self.df_test.copy()

        test_ids = df_test.pop('trip_id')

        #Columns to drop from dataframes
        cols2drop = ['shopper_id', 'fulfillment_model', 'shopping_started_at']

        def dropper(df):
            df.drop(cols2drop, axis=1, inplace=True)

        dropper(df_train)
        dropper(df_test)

        X_train, X_test, y_train = df_train, df_test, self.y_train

        if scale:

            X_train, X_test = scaler(df_train, df_test)

            #Concave function
            y_train = np.sqrt(y_train)

        return X_train, X_test, test_ids, y_train


def scaler(df_train, df_test):
    '''
    Helper function to scale X.
    '''
    cols2scale = ['reorder_factor', 'basket_qty', 'dist_i_cnt', 'dist_dpts_cnt']
    colsnotscale = df_train.columns[~df_train.columns.isin(cols2scale)]

    X_train_cols2scale = df_train.loc[:,cols2scale]
    X_test_cols2scale = df_test.loc[:,cols2scale]

    scaler = StandardScaler().fit(X_train_cols2scale)
    X_train_cols2scale = pd.DataFrame(scaler.transform(X_train_cols2scale), columns = cols2scale)
    X_test_cols2scale = pd.DataFrame(scaler.transform(X_test_cols2scale), columns = cols2scale)

    df_train = pd.concat([df_train.loc[:,colsnotscale], X_train_cols2scale], axis = 1)
    df_test = pd.concat([df_test.loc[:,colsnotscale], X_test_cols2scale], axis = 1)

    return df_train, df_test

if __name__ == '__main__':
    pass
