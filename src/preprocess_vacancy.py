import pandas as pd
from pathlib import Path
import sys
import datetime

PATH = '/Users/danil/Documents/github/jooble'
sys.path.append(str(PATH))

from src.jooble import prepare_csv, preprocess

start = datetime.datetime.now()

# open train dataset
train_features = prepare_csv(path = PATH + '/data/raw/train.tsv')
print('train dataset shape:', train_features.shape)

# fit scaler
prep = preprocess()

prep.get_features_num(train_features)
prep.fit_scaler(train_features)

test_features = prepare_csv(PATH + '/data/raw/test.tsv')
print('test dataset shape:', test_features.shape)

test_features = prep.add_max_feature_index(test_features)
test_features = prep.add_max_feature_2_abs_mean_diff(test_features)
test_features = prep.transform_with_scaler(test_features)

test_features = test_features.rename(columns={i: f'feature_2_stand_{i}' for i in range(1, 1 + prep.features_num)})
test_features = test_features.drop(columns=[0])

# save file
test_features.to_csv(PATH + '/data/processed/test_proc.tsv', sep ='\t', index = False)
end = datetime.datetime.now()
print('process takes:', end - start)
