#!/usr/bin/env python3
import glob
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from xgboost import XGBRegressor


def search_dataset():
    datasets = glob.glob('*.csv')
    assert len(datasets) >= 1, 'No datasets found!'
    if len(datasets) == 1:
        print('Dataset found: {}'.format(datasets[0]))
    else:
        print('Multple datasets found! Using only {} for training/testing.'.format(datasets[0]))
    return datasets[0]


def load_data(filename, na_values):
    df = pd.read_csv(filename, na_values=na_values)
    return df


def preprocess(df):
    features_df = pd.DataFrame()
    target_ser = pd.Series()
    df.dropna(subset=['price'], axis='index', inplace=True)
    numeric_cols_to_use = ['engine-size', 'curb-weight', 'highway-mpg', 
        'horsepower', 'width', 'length', 'normalized-losses', 
        'compression-ratio', 'city-mpg', 'wheel-base', 'peak-rpm', 
        'height', 'stroke', 'bore']
    categorical_cols_to_use = ['make_bmw']
    for col in numeric_cols_to_use:
        if skew(np.abs(df[col]) > 1.0):
            features_df[col] = np.log(1 + df[col])
        else:
            features_df[col] = df[col]
    features_df.fillna(features_df.median(), inplace=True)
    features_df['make_bmw'] = df['make'].apply(lambda x: 1 if x == 'bmw' 
                                                else 0)
    target_ser = df['price']
    return features_df, target_ser


def split(X, y, seed=42)
    return train_test_split(X, y, random_state=seed)


def build_model():
    pipeline = Pipeline(steps=[
        ('polynomial_features', PolynomialFeatures(degree=2)),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.9)),
        ('regressor', XGBRegressor(n_estimators=200, max_depth=2, 
            learning_rate=0.035))
    ])
    return pipeline


def train(model, X, y):
    model.fit(X, y)


def validate(model, X_train, y_train, X_test, y_test):
    print('Training score: {}'.format(X_train, y_train))
    print('Test score: {}'.format(X_test, y_test))


if __name__ == '__main__':
    dataset_filename = search_dataset()
    df = load_data(dataset_filename)
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split(X, y)
    model = build_model()
    train(model, X_train, y_train)
    validate(model, X_train, y_train, X_test, y_test)