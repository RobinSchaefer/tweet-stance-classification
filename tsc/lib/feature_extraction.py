#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
#import tf_sentencepiece

from tsc.lib.helpers import load_fasttext_vectors, load_glove_vectors

def create_fasttext_embeddings(data, vector_path):
    '''
    Create documents embeddings based on pretrained fastText embeddings.

    Function makes use of pretrained fastText word embeddings in order to create
    one vector per text. This is achieved my averaging over the vectors of the
    words used in the text. Dimensions of output embeddings depend on pretrained
    embeddings.

    Reference: https://fasttext.cc/docs/en/crawl-vectors.html

    Input:
    data (list) -- a list of texts
    vector_path (str) -- the path of the pretrained embeddings

    Output:
    embeddings (list) -- the embedding list
    '''

    pretrained_vecs = load_fasttext_vectors(vector_path)

    embeddings = []

    for text in data:
        embedding = np.zeros(len(pretrained_vecs['hello']))
        tokenized_text = text.split()
        c = 0
        for word in tokenized_text:
            try:
                embedding += pretrained_vecs[word]
                c += 1
            except:
                pass
        embedding = embedding/c
        embeddings.append(list(embedding))
    return embeddings

def create_glove_embeddings(data, vector_path):
    '''
    Create documents embeddings based on pretrained GloVe embeddings.

    Function makes use of pretrained GloVe word embeddings in order to create
    one vector per text. This is achieved my averaging over the vectors of the
    words used in the text. Dimensions of output embeddings depend on pretrained
    embeddings.

    Reference: https://nlp.stanford.edu/projects/glove/

    Input:
    data (list) -- a list of texts
    vector_path (str) -- the path of pretrained embeddings

    Output:
    embeddings (list) -- the embedding list
    '''

    pretrained_vecs = load_glove_vectors(vector_path)
    embeddings = []
    for text in data:
        embedding = np.zeros(len(pretrained_vecs['hello']))
        tokenized_text = text.split()
        c = 0
        for word in tokenized_text:
            try:
                embedding += pretrained_vecs[word]
                c += 1
            except:
                pass
        embedding = embedding/c
        embeddings.append(list(embedding))
    return embeddings

def create_ngrams(text, n=2):
    '''
    Creates ngrams.

    Input:
    text (str) -- string of input text
    n (int) -- n (number of 'tokens' per gram)

    Output (list) -- the list of ngrams
    '''
    return [text[i:i+n] for i in range(len(text)-n+1)]

def create_ngrams_from_tokenized_data(data, n=[1]):
    '''
    Convert tokenized data to token ngrams.

    Function converts text to token ngrams (based on n). total_grams is a list
    containing list of ngrams (one per text).

    Input:
    data (list) -- a list of texts
    n (list) -- a list of n (both one and multiple n are allowed)

    Output:
    total_grams (list) -- a list of ngram lists
    '''
    total_grams = []

    if len(n) > 1:
        for text in data:
            grams = []
            for n_ in n:
                grams += [' '.join(gram) for gram in create_ngrams(text.split(), n=n_)]
            total_grams.append(grams)
    else:
        for text in data:
            grams = [' '.join(gram) for gram in create_ngrams(text.split(), n=n[0])]
            total_grams.append(grams)

    return total_grams

def create_stance_features(json, stance_categories):
    '''
    Create stance features based on explicit stance annotations in json data.

    Function outputs feature vector containing explicit stance classification.
    The stances have been annotated according to three groups (against: 0;
    favor: 1; none: 2). Length of feature vector per x: len(stance_categories)

    Input:
    json (list) -- a list of dicts (json objects)
    stance_categories (list) -- a list of stance names

    Output:
    all_vectors (ndarray) -- a vector containing encoded stance labels per text
    '''
    all_vectors = []

    for entry in json:
        vector = []
        for stance_category in stance_categories:
            if entry[stance_category] == 'against':
                vector.append(0)
            elif entry[stance_category] == 'favor':
                vector.append(1)
            elif entry[stance_category] == 'none':
                vector.append(2)
            else:
                vector.append(-1)
        all_vectors.append(vector)

    return np.array(all_vectors)

def create_use_embeddings(data):
    '''
    Create documents embeddings based on the en-de Universal Sentence Encoder
    (USE). Function makes use of the pretrained en-de USE model in order to
    create one vector per text. The model has been trained on sentence and can
    be successfully applied to short texts as well. Embeddings are
    512-dimensional.

    Reference: https://tfhub.dev/google/universal-sentence-encoder-xling/en-de/1

    Input:
    data (list) -- a list of texts

    Output:
    embeddings (ndarray) -- the embedding array
    '''

    graph = tf.Graph()
    with graph.as_default():
        text_input = tf.compat.v1.placeholder(dtype=tf.string, shape = [None])
        encoder = hub.Module('https://tfhub.dev/google/universal-sentence-encoder-xling/en-de/1')
        embedded_text = encoder(text_input)
        init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    graph.finalize()

    session = tf.Session(graph = graph)
    session.run(init_op)

    embeddings = []
    for text in data:
        embeddings.append(session.run(embedded_text, feed_dict = {text_input: [text]}))

    return embeddings

def create_vocabulary(data, limit = None):
    '''
    Create vocabulary.

    Function creates two outputs: 1. the vocabulary set; 2. a vocab-index mapping
    where the respective token functions as the key and the index is the value.

    Input:
    data (list) -- a list of texts
    limit (int) -- upper threshold of vocabulary

    Output:
    vocab (set) -- the vocabulary set
    vocab_index_mapping (dict) -- key: token; value: index

    '''
    vocab = set()
    vocab_index_mapping = {}
    cnt = Counter()

    for entry in data:
        for word in entry:
                cnt[word] += 1

    #create vocab based on limit
    if isinstance(limit, int):
        vocab.update([token for token, count in cnt.most_common(limit)])
    elif limit is None:
        vocab.update([token for token in cnt])

    #create vocab_index_mapping based on vocab
    for i, v in enumerate(vocab):
        vocab_index_mapping[v] = i

    return vocab, vocab_index_mapping

def create_word_embeddings(data, model_type = 'use', vector_path=''):
    '''
    Create embeddings.

    Function creates word or sentence embeddings dependent on the model_type to
    be used.

    Input:
    data (list) -- a list of texts
    model_type (str) -- the model type declaration ('fasttext', 'glove' or 'use')
    vector_path (str) -- the path of the pretrained vectors (only for 'fasttext' or 'glove')

    Output:
    embeddings (ndarray) -- the embedding array
    '''

    if model_type == 'fasttext':
        embeddings = create_fasttext_embeddings(data, vector_path=vector_path)
    elif model_type == 'glove200' or model_type == 'glove300':
        embeddings = create_glove_embeddings(data, vector_path=vector_path)
    elif model_type == 'use':
        embeddings = create_use_embeddings(data)

    return embeddings

def vectorize_ngrams(ngrams, vocab):
    '''
    Create feature vector from ngrams.

    Function converts ngrams to binary bag-of-words representation.

    Input:
    ngrams (list) -- a list of ngram lists
    vocab (set) -- the vocabulary set

    Output (ndarray) -- the ngram array
    '''
    X = []
    for grams in ngrams:
        x = []
        for v in vocab.keys():
            if v in grams:
                x.append(1)
            else:
                x.append(0)
        X.append(x)
    return np.asarray(X)
