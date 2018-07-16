#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
import pandas as pd
from sklearn.model_selection import learning_curve
import itertools
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import requests
import json
from pandas.io.json import json_normalize

from sklearn import preprocessing

class Dataset:

    def __init__(self):
        pass

    @staticmethod
    def read_data(filename):
        docs, labels = [], []
        for line in open(filename, 'r').read().split('\n'):
            e = line.split('|')
            if len(e) == 2:
                body, label = e[0], e[1]
                
                if label not in ['POS', 'NEG', 'NEU']:
                    print(line)
                
                docs.append(body)
                labels.append(label)

        return docs, labels

    def extract_features(self, filename, features):
        docs, labels = Dataset.read_data(filename)

        url = 'http://147.91.183.8:12345/'

        docs_features = []
        label_dict = {'POS': 1, 'NEG': -1, 'NEU': 0}

        for i in range(len(docs)):
            doc = docs[i]
            label = label_dict[labels[i]]
            data = {'data': doc, 'lang_list': ['sr', 'en']}
            feature_dict = {'class': label}

            for feature in features:
                r = requests.post(url + feature, json=data)
                
                feature_dict.update(r.json())

            docs_features.append(feature_dict)

        return json_normalize(docs_features)

    @staticmethod
    def draw_class_dist(objects, dataset_dist, title, filename):

        plt.figure()

        y_pos = np.arange(len(objects))

        p1 = plt.bar(y_pos, dataset_dist, align='center', alpha=1, hatch='')

        plt.xticks(y_pos, objects)
        plt.ylabel('Number of samples')
        plt.title(title)
        plt.savefig('imgs/' + filename)

    @staticmethod
    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), feature_set='lex'):
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        plt.savefig('imgs/' + 'learning-curve-%s.png' % feature_set)
        
    def plot_confusion_matrix(cm, 
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues,feature_set='lex'):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        
        classes = ('negative', 'neutral', 'positive')
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        #plt.show()
        plt.savefig('imgs/' + 'confusion_matrix-%s.png' % feature_set)

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print('Usage: python main.py path_to_csv_or_txt')
        print('Usage: python main.py sms lex-styl-emot-abb')
        exit(1)
        
    feature_set = sys.argv[1][:-4]    
        
    # this means that we want to extract features
    if len(sys.argv) == 3:
        feature_set = sys.argv[2]
        typ = sys.argv[1]
        
        # create dataset frame
        dataset = Dataset()
        
        features = []
        if 'lex' in feature_set:
            features.append('char_based_features')
            features.append('word_based_features')
        if 'abb' in feature_set:
            features.append('abbreviation_features')
        if 'styl' in feature_set:
            features.append('stylistic_features')
        if 'emot' in feature_set:
            features.append('emoticon_features')

        df = dataset.extract_features('dataset/%s.txt' % typ, features)
        df.to_csv('dataset/%s-%s.csv' % (typ, feature_set))
        
        exit(1)

    
    # import data
    df = pd.read_csv(sys.argv[1])
    
    # shuffle dataset (just in case)
    df = shuffle(df)

    # labels
    y = np.array(df['class'])
    
    # samples
    df = df.ix[:, df.columns != 'class']
    # normalize data
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)
    X = np.array(df_normalized)

    print(np.shape(X), np.shape(y))

    clf = LinearSVC(C=0.1, penalty='l2')
    # 1 - positive class
    accuracy = []
    f_score_pos, f_score_neg, f_score_neu = [], [], []
    recall_pos, recall_neg, recall_neu = [], [], []
    precision_pos, precision_neg, precision_neu = [], [], []
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        
        y_pred = clf.fit(X[train], y[train]).predict(X[test])

        print(classification_report(y_pred=y_pred, y_true=y[test]))
        
        accuracy.append(accuracy_score(y[test], y_pred))
        
        recall_pos.append(recall_score(y[test], y_pred, average='weighted', labels=[1]))
        precision_pos.append(precision_score(y[test], y_pred, average='weighted', labels=[1]))
        f_score_pos.append(f1_score(y[test], y_pred, average='weighted', labels=[1]))
        
        recall_neu.append(recall_score(y[test], y_pred, average='weighted', labels=[0]))
        precision_neu.append(precision_score(y[test], y_pred, average='weighted', labels=[0]))
        f_score_neu.append(f1_score(y[test], y_pred, average='weighted', labels=[0]))
        
        recall_neg.append(recall_score(y[test], y_pred, average='weighted', labels=[-1]))
        precision_neg.append(precision_score(y[test], y_pred, average='weighted', labels=[-1]))
        f_score_neg.append(f1_score(y[test], y_pred, average='weighted', labels=[-1]))
      
    print('\\hline')
    print('Fold&ACC&R (pos)&P (pos)&F (pos)&R (neg)&P (neg)&F (neg)&R (neu)&P (neu)&F (neu)\\\\\\hline')
    for i in range(5):
        print('%d&%.3f&%.3f&%.3f&%.3f&%.3f&%.3f&%.3f&%.3f&%.3f&%.3f\\\\\\hline' % ((i + 1, accuracy[i], recall_pos[i], precision_pos[i], f_score_pos[i], recall_neg[i], precision_neg[i], f_score_neg[i], recall_neu[i], precision_neu[i], f_score_neu[i])))
    print('\\hline')
    print('%s&%.3f&%.3f&%.3f&%.3f&%.3f&%.3f&%.3f&%.3f&%.3f&%.3f\\\\\\hline' % ('Average', np.mean(accuracy), np.mean(recall_pos), np.mean(precision_pos), np.mean(f_score_pos), np.mean(recall_neg), np.mean(precision_neg), np.mean(f_score_neg), np.mean(recall_neu), np.mean(precision_neu), np.mean(f_score_neu)))
    
    # train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print(np.shape(X_train), np.shape(y_train), np.shape(X_test), np.shape(y_test))
    
    # visualizing labels distribution
    classes = ('negative', 'positive', 'neutral')
    
    neg = len([l for l in y if l == -1])
    pos = len([l for l in y if l == 1])
    neu = len([l for l in y if l == 0])
    dataset_dist = [neg, pos, neu]
    
    # plot learning curve
    Dataset.plot_learning_curve(clf, 'Learning curve for LinearSVC(C=0.1, penalty="l2")', X, y, (0.85, 1.01), cv=5, n_jobs=4, feature_set=feature_set)

    #print(classes, dataset_dist)
    Dataset.draw_class_dist(classes, dataset_dist, 'Labels Distribution in the Dataset', 'imgs/class_dist.png')
    
    # fitting model
    clf.fit(X_train, y_train)
    # predicting labels
    y_pred = clf.predict(X_test)
    
    # plot confusion matrix
    Dataset.plot_confusion_matrix(confusion_matrix(y_test, y_pred), title='Non-normalized Confusion Matrix', feature_set=feature_set)
