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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
import itertools
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

def draw_class_dist(objects, dataset_dist, train_dist, title, filename):
    
    plt.figure()
    
    y_pos = np.arange(len(objects))

    p1 = plt.bar(y_pos, dataset_dist, align='center', alpha=1, hatch='')
    p2 = plt.bar(y_pos, train_dist, align='center', alpha=0.8, hatch='+')
    
    plt.legend((p1, p2), ('Whole Dataset', 'Training Set'))
    
    plt.xticks(y_pos, objects)
    plt.ylabel('Number of samples')
    plt.title(title)
    plt.savefig(filename)
    
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
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
    plt.savefig('learning-curve.png')
    
def plot_confusion_matrix(cm, 
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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
    plt.savefig('confusion_matrix.png')
    
def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857 
    By HYRY
    '''
    pc.update_scalarmappable()
    ax = pc.get_axes()
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857 
    - https://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()    
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    # resize 
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))

def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857 
    '''
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        label = 'neutral'
        if t[0] == '-1':
            label = 'negative'
        elif t[0] == '1':
            label = 'positive'
        class_names.append(label)
        plotMat.append(v)

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)
    
    plt.savefig('test_plot_classif_report.png', dpi=200, format='png', bbox_inches='tight')


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
    
    # shuffle dataset (just in case)
    df = shuffle(df)

    # samples and labels
    y = np.array(df['class'])
    X = np.array(df.ix[:, df.columns != 'class'])
    print(np.shape(X), np.shape(y))
    
    clf = LinearSVC(C=0.1, penalty='l2')
    
    # 1 - positive class
    accuracy = []
    f_score = []
    recall = []
    precision = []
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        
        y_pred = clf.fit(X[train], y[train]).predict(X[test])
        
        accuracy.append(accuracy_score(y[test], y_pred))
        recall.append(recall_score(y[test], y_pred, average='weighted', labels=[0]))
        precision.append(precision_score(y[test], y_pred, average='weighted', labels=[0]))
        f_score.append(f1_score(y[test], y_pred, average='weighted', labels=[0]))
      
    print('\\hline')
    print('Fold&Accuracy&Recall&Precision&F-score\\\\\\hline')
    for i in range(5):
        print('%d&%.3f&%.3f&%.3f&%.3f\\\\\\hline' % ((i + 1, accuracy[i], recall[i], precision[i], f_score[i])))
    print('\\hline')
    print('%s&%.3f&%.3f&%.3f&%.3f\\\\\\hline' % ('Average', np.mean(accuracy), np.mean(recall), np.mean(precision), np.mean(f_score)))

    # train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print(np.shape(X_train), np.shape(y_train), np.shape(X_test), np.shape(y_test))
    
    # visualizing labels distribution
    classes = ('negative', 'positive', 'neutral')
    
    neg = len(df[df["class"] == -1])
    pos = len(df[df["class"] == 1])
    neu = len(df[df["class"] == 0])
    dataset_dist = [neg, pos, neu]
    
    neg = len([y for y in y_train if y == -1])
    pos = len([y for y in y_train if y == 1])
    neu = len([y for y in y_train if y == 0])
    train_dist = [neg, pos, neu]

    draw_class_dist(classes, dataset_dist, train_dist, 'Labels Distribution in the Dataset', 'class_dist.png')
    
    clf = LinearSVC(C=0.1, penalty='l2')
    # fitting model
    clf.fit(X_train, y_train)
    # predicting labels
    y_pred = clf.predict(X_test)
    
    # plot classification report
    plot_classification_report(classification_report(y_test, y_pred))
    
    # CV score
    k = 5
    scores = cross_val_score(clf, X, y, cv=k, scoring='accuracy')
    print("Accuracy Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(scores), np.std(scores), np.min(scores), np.max(scores)))
    
    # plot learning curve
    plot_learning_curve(clf, 'Learning curve for LinearSVC(C=0.1, penalty="l2")', X, y, (0.85, 1.01), cv=k, n_jobs=4)
    
    # plot confusion matrix
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), title='Non-normalized Confusion Matrix')
