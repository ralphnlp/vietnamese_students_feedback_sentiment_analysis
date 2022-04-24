import os
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import numpy as np


def preprocessing_texts(texts, split = True):
    preproceesed_texts, max_len = [], 0
    for text in texts:
        tokens = ViTokenizer.tokenize(text).split()
        tokens = [token.lower() for token in tokens if token.isalpha() or '_' in token]
        if max_len < len(tokens):   max_len = len(tokens)
        if split == True:
            preproceesed_text = tokens
        else:
            preproceesed_text = ' '.join(tokens)
        preproceesed_texts.append(preproceesed_text)
    return preproceesed_texts, max_len


def load_texts(dir, split=True):
    with open(os.path.join(dir, 'sents.txt'), 'r') as file:
        texts = file.readlines()  
    with open(os.path.join(dir, 'sentiments.txt'), 'r') as file:
        sentiments = file.readlines()
    with open(os.path.join(dir, 'topics.txt'), 'r') as file:
        topics = file.readlines()

    preprocessed_texts, max_len = preprocessing_texts(texts, split)
    return (preprocessed_texts, max_len), sentiments, topics


if __name__=='__main__':

    train_dir = './uit_vsfc_dataset/train/'
    dev_dir = './uit_vsfc_dataset/dev/'
    test_dir = './uit_vsfc_dataset/test/'

    sentiments_dic = {'0': 'negative', '1': 'neutral', '2': 'positive'}
    topics_dic = {'0': 'lecturer', '1': 'training_program', '2': 'facility', '3': 'others'}

    (train_sents, train_max_len), train_sentiments, train_topics = load_texts(train_dir, split=False)
    (dev_sents, _), dev_sentiments, dev_topics = load_texts(dev_dir)
    (test_sents, _), test_sentiments, test_topics = load_texts(test_dir, split=False)
    print('the number of the train = {}, dev = {}, test = {}'.format(len(train_sents), len(dev_sents), len(test_sents)))
    print('hist sentimens = {}, hist topics = {}'.format(np.bincount(train_sentiments), np.bincount(train_topics)))

    
    tfidfer = TfidfVectorizer()
    tfidf_train_sents = tfidfer.fit_transform(train_sents)
    vocab = tfidfer.vocabulary_
    size_vocab = len(vocab)
    print('no the vocabulary = {}'.format(size_vocab))

    clf = MultinomialNB()
    clf.fit(tfidf_train_sents, train_sentiments)

    tfidf_test_sents = tfidfer.transform(test_sents)
    predicts = clf.predict(tfidf_test_sents)
    print(classification_report(test_sentiments, predicts))




