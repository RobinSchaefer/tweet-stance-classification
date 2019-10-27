#!/usr/bin/env python
# -*- coding: utf-8 -*-

from helpers import get_files, apply_tweet_tokenizer

if __name__ == "__main__":
    tokenizer_path = '/home/robin/Documents/software/ark-tweet-nlp-0.3.2'
    txt_path = '/home/robin/Documents/phd/tweet-stance-classification/corpus_txt'
    output_path = '/home/robin/Documents/phd/tweet-stance-classification/corpus_tokenized'
    files = get_files(txt_path)
    apply_tweet_tokenizer(files, tokenizer_path, output_path)
