import pandas as pd
import numpy as np
from numba import jit
import multiprocessing as mp
from pathlib import Path
import sys

PATH = '/Users/danil/Documents/github/jooble'
sys.path.append(str(PATH))


def extract_features(df: pd.DataFrame, column_with_features: str = 'features') -> pd.DataFrame:
    '''
    
    expand features from column 'column_with_features' to separate columns
    
    Parameters
    -------------
    df: pd.DataFrame
        dataframe with columns ['id_job', column_with_features]
        
    return 
    -------------
        dataframe with expanded features to separate columns
    '''
    
    parsed_features = df[column_with_features].str.split(',',
                                               expand=True).astype(np.int16)
    parsed_features.insert(0, 'id_job', df['id_job'])
    return parsed_features


def prepare_csv(path: str, sep: str = '\t', chunksize: int = 1000) -> pd.DataFrame:
    '''
    
    read tsv and convert features to separate columns
    
    Parameters
    -------------
    path: str
        path to .tsv file
        
    sep: str = '\t'
        file separator
        
    chunksize: int = 1000
        number of rows to read from file in batch 
        
    return 
    -------------
        dataframe with expanded features to separate columns
    '''

    batch_reader = pd.read_csv(filepath_or_buffer=path,
                               sep=sep,
                               chunksize=chunksize,
                               engine='c')

    with mp.Pool(mp.cpu_count()) as pool:
        res = pool.map(extract_features, batch_reader)
        df = pd.concat(res)

    return df


@jit(nopython=True)
def fit_z_score_normalizator(X: np.ndarray, features_num: int) -> list:
    '''
    extract column statistic as mean and std, and return them in list
    '''
    
    X_t = X.T
    means = []
    sigmas = []
    
    for i in range(features_num):
        means.append(np.mean(X_t[i]))
        sigmas.append(np.std(X_t[i]))
        
    return means, sigmas


@jit(nopython=True)
def fit_min_max_scaler(X: np.ndarray, features_num: int) -> list:
    # this one I made for example how to add new scaler
    '''
    extract column statistic as min and max, and return them in list
    '''
    
    X_t = X.T
    min_v = []
    max_v = []
    
    for i in range(features_num):
        min_v.append(np.min(X_t[i]))
        max_v.append(np.max(X_t[i]))
        
    return min_v, max_v


class preprocess:
    '''
    class for preprocessing and ading new features to dataframe with jobs and their features
    
    Parameters
    ----------
    scaler: str
        type of scaler 
        ['z-score', 'min-max-scaler']
    
    Attributes
    ----------
    factor: int
        factor of feature set
    
    features_num: int
        number of features for current factor
        
    mean, sigma, min, max : float
        params for scaler based on scaler and fitted on trained dataset with fit_scaler
    '''
    
    def __init__(self, scaler='z-score'):

        self.scaler = scaler

    def get_features_num(self, X):
        '''
        parse factor and define features_num
        '''
        
        self.factor = X[0].value_counts().index[0]
        if self.factor == 2:
            self.features_num = 256
        # in case of selecting special dtype, further we can select only specific column type
        # self.dtypes = int
        # or add selector for dataset filter
        # here we can add new factor check for features selection or checks
        # or redefine normalization
        # self.normalization = 'z-score'

    def fit_scaler(self, X):
        '''
        fit scaler on train dataset and save params
        '''
        
        X = X.iloc[:, 2:2 + self.features_num].values

        if self.scaler == 'z-score':
            self.mean, self.sigma = fit_z_score_normalizator(X, self.features_num)
        elif self.scaler == 'min-max-scaler':
            self.min, self.max = fit_min_max_scaler(X, self.features_num)
        return

    def transform_with_scaler(self, X):
        '''
        transform test dataset with scaler with predefined params from fit_scaler
        '''
        
        X_to_transform = X.iloc[:, 2:2 + self.
                                features_num]  #.select_dtypes(include = self.dtypes)

        if self.scaler == 'z-score':
            X_to_transform = (X_to_transform - self.mean) / self.sigma

        elif self.scaler == 'min-max-scaler':
            X_to_transform = (X_to_transform - self.min) / (self.max -
                                                            self.min)

        X.iloc[:, 2:2 + self.features_num] = X_to_transform

        return X

    def add_max_feature_index(self, X):
        '''
        add feature max_feature_2_index. it is index of feature that has max value for id_job
        '''
        
        X['max_feature_2_index'] = X.iloc[:, 2:2 +
                                          self.features_num].values.argmax(
                                              axis=1)

        return X

    def add_max_feature_2_abs_mean_diff(self, X):
        '''
        add feature max_feature_2_abs_mean_diff. it is difference between max value among job features
        and mean value for this feature among all train dataset
        '''
        
        max_values = X.iloc[:, 2:2 + self.features_num].values.max(axis=1)
        mean_values = X.iloc[:, 2:2 + self.features_num].values.mean(axis=0)

        if 'max_feature_2_index' in X.columns:
            indexes = X['max_feature_2_index']
        else:
            indexes = self.add_max_feature_index(X)['max_feature_2_index']

        X['max_feature_2_abs_mean_diff'] = np.abs(max_values -
                                                  mean_values[indexes])

        return X