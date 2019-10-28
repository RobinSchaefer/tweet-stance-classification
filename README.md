# Tweet Stance Classification

This code can be used to reproduce the results of our stance classification
system for English tweets (based on Wojatzki & Zesch 2016).

This work has been published in:

Robin Schaefer and Manfred Stede. Improving Implicit Stance Classification in
Tweets Using Word and Sentence Embeddings. In Proceedings of the 42nd German
Conference on Artificial Intelligence (KI 2019). Kassel, Germany, 2019.

## Installation

This code was written for Python 3.

Set up a virtualenv and run:

<code>pip3 install -r requirements.txt</code>

This code makes use of several pretrained embeddings. Download and store them in the respective directory:

fastText vectors (in /tsc/embeddings):
https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip

GloVe vectors (200d) (in /tsc/embedings):
https://nlp.stanford.edu/data/glove.twitter.27B.zip

GloVe vectors (300d) (in /tsc/embeddings):
https://nlp.stanford.edu/data/glove.42B.300d.zip

The tweet data can be downloaded from here: https://github.com/muchafel/AtheismStanceCorpus

We already reshaped, tokenized and stored it in /tsc/data.

## Usage

For training, testing and evaluating a model, run:

<code>python -m tsc</code>

You can modify the procedure by setting options for e.g. the preprocessing or the feature type.

For turning the data to lowercase, run:

<code>python -m tsc -p l</code>

For training the model with fastText vectors, run:

<code>python -m tsc -f fasttext</code>

For a full description of options and parameters, run:

<code>python -m tsc --help</code>

## License

MIT License
