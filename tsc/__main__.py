#!/usr/bin/env python
# -*- coding: utf-8 -*-

### MODULE IMPORTS ###
import click
import numpy as np
from sklearn.model_selection import StratifiedKFold
from nltk.corpus import stopwords
import os
from statistics import mean
import warnings
warnings.filterwarnings("ignore")


from tsc.lib.classification import SVMClassifier, evaluate_model, create_evaluation_table
from tsc.lib.feature_extraction import create_ngrams_from_tokenized_data, create_stance_features, create_vocabulary, create_word_embeddings, vectorize_ngrams
from tsc.lib.preprocessing import apply_preprocessing_pipeline, encode_labels
from tsc.lib.helpers import convert_corpus_to_json, convert_preprocessing_encodings, convert_n_str, get_files, load_corpus_as_JSON, load_tokenized_files, reshape_array, split_data
################################################################################

### PATH DEFINITIONS ###
cwd = os.getcwd()

corpus_path = cwd + '/tsc/data/atheism_stance_corpus.txt'
json_path = cwd + '/tsc/data/atheism_stance_corpus.json'
tokenized_path = cwd + '/tsc/data/corpus_tokenized'
fasttext_vectors_path = cwd + '/tsc/embeddings/wiki-news-300d-1M.vec'
glove_vectors_path =  cwd + '/tsc/embeddings/glove.42B.300d.txt'

### PREPROCESSING PARAMETERS ###
stop_list = stopwords.words('english')
stop_list += ['.', ',']
ignored_punctuation = ['#']

### FEATURE EXTRACTION PARAMETERS ###
target = 'debateStancePolarity'


################################################################################

#Stance categories
stance_categories = ['secularismPolarity', 'religiousFreedomPolarity',
                     'freethinkingPolarity', 'noEvidencePolarity',
                     'supernaturalPowerPolarity', 'christianityPolarity',
                     'lifeAfterDeathPolarity', 'usaPolarity', 'islamPolarity',
                     'conservatismPolarity', 'sameSexMarriagePolarity']

@click.command()
@click.option('--preprocessing', '-p', default='l', type=str, show_default=True, help='Preprocessing steps: l (lowercase), s (stopwords), p (punctuation), st (stemming)')
@click.option('--feature', '-f', default='ngram', type=str, show_default=True, help='Feature type: ngram, stance, fasttext, glove200, glove300, use')
@click.option('--target', '-t', default='debateStancePolarity', type=str, show_default=True, help='Stance target: debateStancePolarity, secularismPolarity, religiousFreedomPolarity, \
                                                                               freethinkingPolarity, noEvidencePolarity, supernaturalPowerPolarity, christianityPolarity, \
                                                                               lifeAfterDeathPolarity, usaPolarity, islamPolarity, conservatismPolarity, sameSexMarriagePolarity.')
@click.option('-n', default='1', show_default=True, type=str, help='n in n-gram.')
@click.option('-k', default=10, type=int, show_default=True, help='k in k-fold cross-validation.')
@click.option('--metric', '-m', default='micro', type=str, show_default=True, help='Metric averaging type: micro, macro, weighted.')
@click.option('--kernel', default='linear', type=str, show_default=True, help='SVM kernel.')
@click.option('-c', default=1, type=int, show_default=True, help='SVM C hyperparameter.')


def main(preprocessing, feature, target, n, k, metric, kernel, c):

    if not os.path.isfile(json_path):
        convert_corpus_to_json(corpus_path, json_path)

    print(80*'#')
    print('\nFeature: {}'.format(feature))

    # load data
    json = load_corpus_as_JSON(json_path)

    files = get_files(tokenized_path, file_content='tokenized')
    data = load_tokenized_files(files)

    #apply preprocessing
    print('\nPreprocessing:')
    preprocessing = convert_preprocessing_encodings(preprocessing)
    preprocessed_data = apply_preprocessing_pipeline(data, pipeline=preprocessing, stopwords=stop_list, ignored_punctuation=ignored_punctuation, tags=False)

    #apply feature extraction
    if feature == 'ngram':
        n = convert_n_str(n)
        ngrams = create_ngrams_from_tokenized_data(preprocessed_data, n=n)
        vocab, vocab_index_mapping = create_vocabulary(ngrams, limit=None)
        X = vectorize_ngrams(ngrams, vocab=vocab_index_mapping)
    elif feature == 'stance':
        X = create_stance_features(json, stance_categories=stance_categories)
    elif feature == 'fasttext':
        X = create_word_embeddings(preprocessed_data, model_type=feature, vector_path=fasttext_vectors_path)
        X = reshape_array(X)
    elif feature == 'glove300':
        X = create_word_embeddings(preprocessed_data, model_type=feature, vector_path=glove_vectors_path)
        X = reshape_array(X)
    elif feature == 'use':
        X = create_word_embeddings(preprocessed_data, model_type=feature)
        X = reshape_array(X)

    #extract labels
    y = encode_labels(json, target=target)

    ### TRAIN, TEST AND EVALUATE MODEL ###
    # with k-fold cross validation
    if k > 1:
        skf = StratifiedKFold(n_splits=k, shuffle=False)
        accuracies= []
        f1s = []
        precisions = []
        recalls = []
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = SVMClassifier(X_train, y_train, kernel=kernel, C=c)
            acc, f1, prec, recall = evaluate_model(clf, X_test, y_test, average=metric)
            accuracies.append(acc)
            f1s.append(f1)
            precisions.append(prec)
            recalls.append(recall)
        print('\nModel evaluation for {} rounds:'.format(k))
        print(create_evaluation_table(accuracies, f1s, precisions, recalls))
        print('\nModel evaluation after {}-fold CV:\nAccuracy score: {}\nF1 score: {}\nPrecision score: {}\nRecall score: {}\n'.format(k, round(mean(accuracies),2),round(mean(f1s),2),round(mean(precisions),2),round(mean(recalls),2)))
        print(80*'#')
    # without cross validation
    else:
        #split data
        X_train, X_test, y_train, y_test = split_data(X,y,test_size=0.2)

        #run classifier
        clf = SVMClassifier(X_train, y_train, kernel=kernel, C=c)

        #evaluation of test data
        acc, f1, prec, recall = evaluate_model(clf, X_test, y_test, average=metric)


if __name__ == '__main__':
    main()
