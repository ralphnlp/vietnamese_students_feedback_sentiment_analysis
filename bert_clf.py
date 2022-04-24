import pickle
import os
from sklearn.metrics import classification_report
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP
from tqdm import tqdm


rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
def preprocess_sentences(sentences):

    preprocessed_sentences = []
    for sentence in tqdm(sentences):
        tokenized_sentence = rdrsegmenter.tokenize(sentence)[0]
        preprocessed_sentence = ' '.join(tokenized_sentence)
        preprocessed_sentences.append(preprocessed_sentence)
    return preprocessed_sentences


def load_texts(dir):

    with open(os.path.join(dir, 'sents.txt'), 'r') as file:
        texts = file.readlines()  
    with open(os.path.join(dir, 'sentiments.txt'), 'r') as file:
        sentiments = file.readlines()
    with open(os.path.join(dir, 'topics.txt'), 'r') as file:
        topics = file.readlines()

    tokenized_sentences = preprocess_sentences(texts)
    return tokenized_sentences, sentiments, topics


def str2int(y):
    for i in range(len(y)):
        y[i] = int(y[i][0])
    y = torch.as_tensor(y, dtype=torch.int8)
    return y


phobert = AutoModel.from_pretrained("vinai/phobert-base") 
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
def extract_sentences(sentences):

    def extract(sentence):
        input_ids = torch.tensor([tokenizer.encode(sentence)])
        with torch.no_grad():
            feature = phobert(input_ids)[0][0][0]  # Models outputs are now tuples
            return feature
    
    features = torch.zeros((1, 768))
    for sentence in tqdm(sentences):
        feature = extract(sentence)
        features = torch.concat([features, feature.reshape(1, -1)], dim=0)
    return features[1:]


class SVC(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(768, 3)
        self.softmax = torch.nn.Softmax()

    def _to_categorical(self, y):
        n_samples = y.shape[0]
        n_classes = 3
        onehot_y = torch.zeros((n_samples, n_classes)).cuda()
        for i, label in enumerate(y):
            onehot_y[i][label] = 1
        return onehot_y


    def _forward(self, X):
        outputs = self.linear1(X)
        outputs = self.softmax(outputs)
        return outputs


    def fit(self, X, y, epochs, lr):

        onehot_y = self._to_categorical(y)
        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in tqdm(range(epochs)):
            y_hat = self._forward(X)
            loss = criterion(y_hat, onehot_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch {epoch} th -> loss = {loss}")


    def predict(self, X):
        with torch.no_grad():
            y_hat = self._forward(X)
            y_hat = torch.argmax(y_hat, dim=1).detach().cpu()
            return y_hat
 


if __name__=='__main__':

    train_dir = './uit_vsfc_dataset/train/'
    dev_dir = './uit_vsfc_dataset/dev/'
    test_dir = './uit_vsfc_dataset/test/'

    sentiments_dic = {'0': 'negative', '1': 'neutral', '2': 'positive'}
    topics_dic = {'0': 'lecturer', '1': 'training_program', '2': 'facility', '3': 'others'}

    data_path = './data.pkl'
    if os.path.exists(data_path) == True:
        with open(data_path, 'rb') as file:
            train_features, test_features, train_sentiments, test_sentiments = pickle.load(file)
    else:
        train_sents, train_sentiments, train_topics = load_texts(train_dir)
        dev_sents, dev_sentiments, dev_topics = load_texts(dev_dir)    
        test_sents, test_sentiments, test_topics = load_texts(test_dir)
        print('the number of the train = {}, dev = {}, test = {}'.format(len(train_sents), len(dev_sents), len(test_sents)))
        print('hist sentimens = {}, hist topics = {}'.format(np.bincount(train_sentiments), np.bincount(train_topics)))

        train_features = extract_sentences(train_sents)
        train_sentiments = str2int(train_sentiments)
        test_features = extract_sentences(test_sents)
        test_sentiments = str2int(test_sentiments)

        with open(data_path, 'wb') as file:
            pickle.dump([train_features, test_features, train_sentiments, test_sentiments], file)


    train_features = train_features.cuda()
    test_features = test_features.cuda()
    train_sentiments = train_sentiments.cuda()
    clf = SVC().cuda()
    clf.fit(train_features, train_sentiments, epochs=100, lr=0.5)
    predicts = clf.predict(test_features)
    print(classification_report(test_sentiments, predicts))
