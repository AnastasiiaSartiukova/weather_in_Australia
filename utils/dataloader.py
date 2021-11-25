import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        print(self.dataset.columns)
        self.dataset.drop('Date', axis=1, inplace=True)  # drop the original Date variable
        self.dataset.drop('Location', axis=1, inplace=True)  # drop the original Date variable
        categorical = [col for col in self.dataset.columns if self.dataset[col].dtypes == 'O']  # find categorical variables
        numerical = [col for col in self.dataset.columns if self.dataset[col].dtypes != 'O']  # find numerical variables

        # impute missing values in with respective column median
        for col in numerical:
            col_median = self.dataset[col].median()
            self.dataset[col].fillna(col_median, inplace=True)

        # ENCODE CATEGORICAL VARIABLES
        le = LabelEncoder()
        for cat in categorical:
            le.fit(self.dataset[cat])
            self.dataset[cat] = le.transform(self.dataset[cat])

        #print(self.dataset[categorical].isnull().sum())  # check missing values in numerical variables in X_train

        #print(self.dataset['RainToday'])
        # impute missing categorical variables with the most frequent value
        self.dataset['WindGustDir'].fillna(self.dataset['WindGustDir'].mode()[0], inplace=True)
        self.dataset['WindDir9am'].fillna(self.dataset['WindDir9am'].mode()[0], inplace=True)
        self.dataset['WindDir3pm'].fillna(self.dataset['WindDir3pm'].mode()[0], inplace=True)
        self.dataset['RainToday'].fillna(self.dataset['RainToday'].mode()[0], inplace=True)

        '''# ENGINEERING OUTLIERS IN NUMERICAL VARIABLES
        outl = ['Rainfall', 'Evaporation', 'WindSpeed9am', 'WindSpeed3pm']
        for o in outl:
            data_mean, data_std = np.mean(self.dataset[o]), np.std(self.dataset[o])
            # identify outliers
            cut_off = data_std * 3
            lower, upper = data_mean - cut_off, data_mean + cut_off
            # identify outliers
            outliers = [x for x in self.dataset[o] if x < lower or x > upper]
            # remove outliers
            outliers_removed = [x for x in self.dataset[o] if lower <= x <= upper]'''



        return self.dataset
