#!/usr/bin/env python
# -*- coding: utf-8 -*-
import string

import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

def apply_preprocessing_pipeline(data, pipeline=[], stopwords=[], ignored_punctuation=[], tags=False):
    '''
    Apply defined preprocessing pipeline.

    Preprocessing steps are to be defined as a list of strings using the following keywords:
    - lowercase: run turn_lowercase
    - stopwords: run remove_stopwords
    - punctuation: run remove_punctuation
    - stemming: run apply_stemmer
    - lemmatizing: run apply_lemmatizer
    Additional input arguments may be needed (see input definition)

    Input:
    data (list) -- a list of texts
    pipeline (list) -- a list of processing steps (see above)
    stopwords (list) -- a list of words to be removed (needed for remove_stopwords)
    ignored_punctuation (list) -- a list of punctuation to be ignore (needed for remove_punctuation)
    tags (boolean) -- declares if data has been POS tagged


    '''
    step = 0

    for task in pipeline:
        if task=='lowercase':
            step += 1
            data = turn_lowercase(data, tags=tags)
            print('Step {}: Data turned to lowercase'.format(step))
            continue
        if task=='stopwords':
            if isinstance(stopwords, list):
                step += 1
                data = remove_stopwords(data, stopwords)
                print('Step {}: Stopwords removed'.format(step))
            continue
        if task=='punctuation':
            if isinstance(ignored_punctuation, list):
                step+=1
                data = remove_punctuation(data, ignored_punctuation=ignored_punctuation)
                print('Step {}: Punctuation removed'.format(step))
            continue
        if task=='stemming':
            step += 1
            data = apply_stemmer(data)
            print('Step {}: Data stemmed (Snowball)'.format(step))
            continue
        if task=='lemmatizing':
            step += 1
            data = apply_lemmatizer(data)
            print('Step {}: Data lemmatized'.format(step))
            continue

    if step == 0:
        print('\nNo preprocessing pipeline defined!')

    return data

def apply_lemmatizer(data):
    '''
    Apply NLTK's WordNet Lemmatizer.

    Input:
    data (list) -- a list of texts

    Output:
    lemmatized_data (list) -- a list of lemmatized texts
    '''

    lemmatized_data = []

    # initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    for text in data:
        lemmatized_text = []
        #split text
        for token in text.split():
            try:
                #lemmatize token
                lemmatized_text.append(lemmatizer.lemmatize(token))
            except:
                #if lemmatization fails keep original token
                lemmatized_text.append(token)
        lemmatized_data.append(' '.join(lemmatized_text))

    return lemmatized_data

def apply_stemmer(data, lang='english'):
    '''
    Apply NLTK's Snowball Stemmer.

    Input:
    data (list) -- a list of texts
    lang (str) -- the used language (default: english)

    Output:
    stemmed_data (list) -- a list of stemmed texts
    '''

    stemmed_data = []

    # initialize stemmer
    stemmer = SnowballStemmer(lang)

    for text in data:
        stemmed_text = []
        # split text
        for token in text.split():
            try:
                # stem token
                stemmed_text.append(stemmer.stem(token))
            except:
                # if stemming fails keep original token
                stemmed_text.append(token)
        stemmed_data.append(' '.join(stemmed_text))

    return stemmed_data

def encode_labels(data, target='debateStancePolarity'):
    '''
    Encode labels.

    Labels get encoded according to the following scheme:
    'against' --> 0
    'favor' --> 1
    'none' --> 2

    Input:
    data (list) -- a list of json objects (dict)
    target (str) -- the json object key containing the target information

    Output:
    stemmed_data (list) -- a list of stemmed texts
    '''

    y = []

    for entry in data:
        if entry[target] == 'against':
            y.append(0)
        elif entry[target] == 'favor':
            y.append(1)
        elif entry[target] == 'none':
            y.append(2)

    # transform encoded label list to np.array
    return np.array(y)

def remove_punctuation(data, ignored_punctuation=[]):
    '''
    Remove punctuation.

    Punctuation will be removed if not contained in ignored_punctuation.

    Input:
    data (list) -- a list of texts
    ignored_punctuation (list) -- a list of punctuation to be ignored

    Output:
    data_without_punctuation (list) -- a list of texts after punctuation removal
    '''

    data_without_punctuation = []

    for text in data:
        # define set of punctuation to be removed
        punctuation = ''.join([p for p in string.punctuation if p not in ignored_punctuation])
        # remove punctuation
        text = text.translate(str.maketrans('', '', punctuation))
        data_without_punctuation.append(text)

    return data_without_punctuation

def remove_stopwords(data, stopwords=[]):
    '''
    Remove stopwords.

    Words to be removed are defined in stopwords.

    Input:
    data (list) -- a list of texts
    stopwords (list) -- a list of words to be removed

    Output:
    data_without_stopwords (list) -- a list of texts after stopwords removal
    '''

    data_without_stopwords = []

    for text in data:
        # remove stopwords from text
        text_without_stopwords = [word for word in text.split() if word not in stopwords]
        data_without_stopwords.append(' '.join(text_without_stopwords))

    return data_without_stopwords

def turn_lowercase(data, tags=False):
    '''
    Turn text to lowercase.

    Function can be used for POS tagged and bare tokens (defined by tags).

    Input:
    data (list) -- a list of texts
    tags (boolean) -- declares if data has been POS tagged

    Output:
    lowercased_data (list) -- a list of lowercased texts
    '''

    lowercased_data = []

    for text in data:
        lowercased_text = []
        # split text
        for token in text.split():
            # for data without POS tags
            if not tags:
                lowercased_text.append(token.lower())
            # for POS tagged data
            else:
                token, tag = token.split('<>')
                lowercased_text.append(token.lower() + '<>' + tag)
        lowercased_data.append(' '.join(lowercased_text))

    return lowercased_data
