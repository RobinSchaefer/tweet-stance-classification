#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import click
import codecs
import json
import os
import shutil

import numpy as np
from sklearn.model_selection import train_test_split


def apply_tweet_tokenizer(files, tokenizer_path, output_path, reset=True):
    '''
    Apply tweet tokenizer.

    This function applies the Tweet NLP tokenizer (http://www.cs.cmu.edu/~ark/TweetNLP/).
    The tokenizer itself is written in Java and applied by twokenize.sh. The shell script
    is run using the os module.

    Input:
    files (list) -- a list of absolute file paths
    tokenizer_path (str) -- the absolute path of the tokenizer
    output_path (str) -- the absolute path for the tokenized output
    '''

    if isinstance(files, list):
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        elif os.path.isdir(output_path) and reset:
            os.remove(output_path)
            print('removed')

        print('START Tokenizing')
        print('Total number of files: {}'.format(len(files)))
        for f in files:
            print('File {}:'.format(f.split('/')[-1]))
            fOut = os.path.abspath(os.path.join(output_path, f.split('/')[-1] + '_tokenized'))
            os.system(tokenizer_path + '/twokenize.sh ' + f + ' > ' + fOut)
        print('END Tokenizing')

def convert_corpus_to_json(corpus_path, json_path):
    '''
    Convert corpus from plain text to json.

    Input:
    corpus_path (str) -- the path to the corpus (text)
    json_path (str) -- the output path to the new corpus file (json)
    '''

    allData = []

    with codecs.open(corpus_path) as txtFile:
        data = txtFile.readlines()

        for ID, row in enumerate(data):
            if ID == 0:
                continue

            splitted = row.split('\t')

            #extract tweet information from line
            id_ = int(splitted[0].split()[0].strip())
            text = ' '.join(splitted[0].split()[1:]).strip()
            debateStance = splitted[1].split(':')[1].strip()
            secularism = splitted[2].split(':')[1].strip()
            religiousFreedom = splitted[3].split(':')[1].strip()
            freethinking = splitted[4].split(':')[1].strip()
            noEvidence = splitted[5].split(':')[1].strip()
            supernaturalPower = splitted[6].split(':')[1].strip()
            christianity = splitted[7].split(':')[1].strip()
            lifeAfterDeath = splitted[8].split(':')[1].strip()
            usa = splitted[9].split(':')[1].strip()
            islam = splitted[10].split(':')[1].strip()
            conservatism = splitted[11].split(':')[1].strip()
            sameSexMarriage = splitted[12].split(':')[1].strip()

            #create json object of tweet
            entry = {'id': id_,
                     'text': text,
                     'debateStancePolarity': debateStance,
                     'secularismPolarity': secularism,
                     'religiousFreedomPolarity': religiousFreedom,
                     'freethinkingPolarity': freethinking,
                     'noEvidencePolarity': noEvidence,
                     'supernaturalPowerPolarity': supernaturalPower,
                     'christianityPolarity': christianity,
                     'lifeAfterDeathPolarity': lifeAfterDeath,
                     'usaPolarity': usa,
                     'islamPolarity': islam,
                     'conservatismPolarity': conservatism,
                     'sameSexMarriagePolarity': sameSexMarriage}
            allData.append(entry)

    #save json in file
    with codecs.open(json_path, 'w') as json_file:
        json.dump(allData, json_file)

def convert_json_to_txt(json_path, txt_path):
    '''
    Convert corpus (json) to text (one tweet per file).

    Input:
    json_path (str) -- the path of the corpus (json)
    txt_path (str) -- the path of the txt files
    '''

    with codecs.open(json_path, mode='r') as f:
        data = json.load(f)

    if not os.path.isdir(txt_path):
        os.mkdir(txt_path)

    for entry in data:
        file_path = os.path.join(txt_path, str(entry['id']))
        with codecs.open(file_path, mode='w') as txt_file:
            text = entry['text']
            txt_file.write(text)

def convert_preprocessing_encodings(preproc_str):
    '''
    '''
    preproc_encoding_mapping = {'l': 'lowercase',
                                'p': 'punctuation',
                                's': 'stopwords',
                                'st': 'stemming',
                                'le': 'lemmatizing'}
    preproc_steps = []

    for s in preproc_str:
        try:
            preproc_steps.append(preproc_encoding_mapping[s])
        except:
            pass 
    return preproc_steps

def convert_n_str(n_str):
    '''
    '''
    return [int(n) for n in n_str]


def load_corpus_as_JSON(json_path, sorted_by='id'):

    with codecs.open(json_path) as json_file:
        data = json.load(json_file)
        data = sorted(data, key=lambda k: k[sorted_by])

    return data

def load_fasttext_vectors(fName):
    """
    A function to load pretrained fasttext vecs as np array.
    """
    with codecs.open(fName, 'r') as f:
        data = {}
        for line in f:
            token = line.split()[0]
            embed = line.split()[1:]
            embed = list(map(float, embed))
            data[token] = embed

        return data

def load_glove_vectors(fName):
    f = codecs.open(fName,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def load_tokenized_files(files):
    '''
    '''
    file_content = []

    for file_ in files:
        with codecs.open(file_, mode='r') as f:
            content = f.read()
            file_content.append(content.split('\t')[0])
    return file_content

def get_files(folder, file_content='tokenized'):

    '''

    '''

    filelist = []

    if file_content == 'raw':
        files = sorted(list(map(int, os.listdir(folder))))
        for f in files:
            abspathFile = os.path.abspath(os.path.join(folder, str(f)))
            filelist.append(abspathFile)
    else:
        files = [f.split('_')[0] for f in os.listdir(folder)]
        files = sorted(list(map(int, files)))
        for f in files:
            abspathFile = os.path.abspath(os.path.join(folder, str(f)+'_'+file_content))
            filelist.append(abspathFile)

    return filelist

def reshape_array(X):
    '''
    Reshape array.

    If array has 3 dimensions, reshape to 2.

    Input:
    X (list) -- a list of embeddings
    X (ndarray) -- the reshape embedding array
    '''
    X = np.array(X)
    if len(X.shape) == 3:
        #X has 3dim; reduce it to 2dim (nx == 1; ny == vector dimensions)
        nsamples, nx, ny = X.shape
        X = X.reshape((nsamples,nx*ny))
    X = np.nan_to_num(X)
    return X

def split_data(X, y, test_size=0.2):
    '''

    '''


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, stratify=y)

    return X_train, X_test, y_train, y_test
