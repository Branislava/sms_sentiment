#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from features_extraction.dataset import Dataset
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd


if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print('Usage: python main.py path_to_csv')
        exit(1)

    '''
    # create dataset frame
    dataset = Dataset(filename=sys.argv[1], verbose=True)
    X, y = dataset.data, dataset.target
    '''
    
    # import data
    df = pd.read_csv(sys.argv[1])
    
    # shuffle dataset
    df = shuffle(df)

    # samples and labels
    y = np.array(df['class'])
    X = np.array(df.ix[:, df.columns != 'class'])
    print(np.shape(X), np.shape(y))

    # train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print(np.shape(X_train), np.shape(y_train), np.shape(X_test), np.shape(y_test))
    
    clf = LinearSVC(C=0.1, penalty='l2')
    # fitting model
    clf.fit(X_train, y_train)
    # predicting labels
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    