#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC, SVC

def SVMClassifier(X, y, kernel='rbf', C=1.0, gamma='auto'):
    '''
    Train a Support Vector Machine classifier.

    Input:
    X (ndarray) -- the array of training data
    y (ndarray) -- the array of training labels
    kernel (string) -- the kernel definition (linear, rbf, poly, sigmoid)
    C (float) -- the C definition
    gamma (float/str) -- the gamma definition (a float value or 'auto')

    Output:
    clf (object) -- the trained model
    '''

    if kernel == 'linear':
        clf = LinearSVC(C=C)
    else:
        clf = SVC(kernel=kernel, C=C, gamma=gamma)

    clf.fit(X,y)

    return clf

def evaluate_model(model, X, y, average='micro'):
    '''
    Evaluate a model.

    Input:
    model (object) -- the trained model
    X (ndarray) -- the array of testing data
    y (ndarray) -- the array of predicted labels
    average (str) -- the average type used

    Output:
    acc (float) -- the model's accuracy
    f1 (float) -- the model's f1 score
    precision (float) -- the model's precision score
    recall (float) -- the model's recall score
    '''

    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average=average)
    precision = precision_score(y, pred, average=average)
    recall = recall_score(y, pred, average=average)
    return acc, f1, precision, recall

def create_evaluation_table(accuracies, f1s, precisions, recalls):
    '''
    '''
    eval_matrix = np.array([np.round(accuracies,2), np.round(f1s,2), np.round(precisions,2), np.round(recalls,2)]).transpose()
    table = pd.DataFrame(eval_matrix, index=list(range(1,11)), columns=['Accuracy', 'F1', 'Precision', 'Recall'])
    return table
