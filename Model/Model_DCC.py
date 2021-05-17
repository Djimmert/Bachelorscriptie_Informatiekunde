import random
import collections
import nltk.classify
from nltk.metrics import precision, recall

from sklearn import svm
from sklearn.svm import LinearSVC
import sklearn.model_selection
import sklearn.metrics

import spacy
from spacy.tokens import Doc
nlp = spacy.load("nl_core_news_lg")

from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer

class Featurizer(TransformerMixin):
    """
    a self constructed featurizer, based on https://drive.google.com/drive/folders/168XkD-lEVEkGj_HKU0_B_IACsWcOBpqB
    """
    def __init__(self):
        self.DELIM=" "
        self.data = []
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        out = [self.return_word(text)
                  for text in X]
        out = [self.capital(text, d)
                  for text, d in zip(X, out)]
        out = [self.first_word(text, d)
                  for text, d in zip(X, out)]
        out = [self.pos(text, d)
                  for text, d in zip(X, out)]
        return out

    def return_word(self, text):
        """
        Sets the word as a feature
        """
        d = {}
        d[text[0].text] = 1
        return d
    
    def capital(self, text, d):
        """
        Sets wether the word starts with a capital letter or not as a feature
        """
        if text[0].text.istitle():
            d["capital"] = 1
        return d
    
    def first_word(self, text, d):
        """
        Sets wether the word is the first word of a sentence or not as a feature
        """
        if text[0].i == 0:
            d["first_word"] = 1
        return d
        
    def pos(self, text, d):
        """
        Sets the POS-tag of spacy trained on Dutch of the previous and next word as a feature, if applicable
        """
        if text[1]:
            d['b'+text[1].pos_] = 1
        if text[2]:
            d['e'+text[2].pos_] = 1
        return d
    
    def word_length(self, text, d):
        """
        Sets word length as a feature
        """
        d[len(text[0].text)] = 1
        return d

def read_data():
    X_feats = []
    y_feats = []
    with open("gro-ner-train.csv", encoding='utf-8') as f:

        sentence = []
        
        for line in f:
            line = line.rstrip().split(";")
            if line[0] == '':                                   # If the end of a sentence is detected
                
                sentence_nlp = Doc(nlp.vocab, words=sentence)   # Spacy performs POS on the sentence
                
                for token in sentence_nlp:
                    
                    if token.i == 0:
                        try:
                            X_feats.append((token, '', sentence_nlp[token.i+1]))
                        except IndexError:
                            X_feats.append((token, '', ''))
                    
                    else:
                        try:
                            X_feats.append((token, sentence_nlp[token.i-1], sentence_nlp[token.i+1]))
                        except IndexError:
                            X_feats.append((token, sentence_nlp[token.i-1], ''))
                
                sentence = []   # Empties sentence for the next one
            
            elif line[1] == '':             # Some annotations are missing in the data. This fixes some of them.
                y_feats.append('O')
                sentence.append(line[0])
            elif line[1] == '"':            # This fixes the others
                y_feats.append('O')
                sentence.append(';')
            else:
                y_feats.append(line[1])
                sentence.append(line[0])
    
        sentence_nlp = Doc(nlp.vocab, words=sentence)      # Same as before
        
        for token in sentence_nlp:
            
            if token.i == 0:
                try:
                    X_feats.append((token, '', sentence_nlp[token.i+1]))
                except IndexError:
                    X_feats.append((token, '', ''))
            
            else:
                try:
                    X_feats.append((token, sentence_nlp[token.i-1], sentence_nlp[token.i+1]))
                except IndexError:
                    X_feats.append((token, sentence_nlp[token.i-1], ''))
                    
            
    return X_feats, y_feats

def train(X_train, y_train):
    classifier = svm.SVC(kernel='linear').fit(X_train, y_train)

    return classifier


def main():
    X_feats, y_feats = read_data()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_feats, y_feats, test_size=0.2)
    
    featurizer = Featurizer()
    vectorizer = DictVectorizer()

    X_train_dict = featurizer.fit_transform(X_train)
    X_test_dict = featurizer.transform(X_test)

    X_train_ = vectorizer.fit_transform(X_train_dict)
    X_test_ = vectorizer.transform(X_test_dict)

    classifier = train(X_train_, y_train)
