import sys

import random
import collections
import nltk.classify
from nltk.metrics import precision, recall

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import sklearn.model_selection
from sklearn.model_selection import KFold
import sklearn.metrics

import numpy as np

import spacy
from spacy.tokens import Doc
nlp = spacy.load("nl_core_news_lg")

from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer

class Featurizer(TransformerMixin):
    """
    a self constructed featurizer, based on https://drive.google.com/drive/folders/168XkD-lEVEkGj_HKU0_B_IACsWcOBpqB
    """
    def __init__(self, features):
        self.DELIM=" "
        self.data = []
        self.features = features
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        features_word = ["tokens"]
        out = [self.return_word(text)
                  for text in X]
        if "capital" in self.features:
            features_word.append("capital")
            out = [self.capital(text, d)
                      for text, d in zip(X, out)]
        if "first_word" in self.features:
            features_word.append("first_word")
            out = [self.first_word(text, d)
                      for text, d in zip(X, out)]
        if "pos" in self.features:
            features_word.append("pos")
            out = [self.pos(text, d)
                      for text, d in zip(X, out)]
        if "word_length" in self.features:
            features_word.append("word_length")
            out = [self.word_length(text, d)
                      for text, d in zip(X, out)]
        if "surrounding_tokens" in self.features:
            features_word.append("surrounding_tokens")
            out = [self.surrounding_tokens(text, d)
                      for text, d in zip(X, out)]
        print("[Features]: {}".format(", ".join(features_word)))
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
            d['before_pos'] = text[1].pos_
        if text[2]:
            d['after_pos'] = text[2].pos_
        return d
    
    def word_length(self, text, d):
        """
        Sets word length as a feature
        """
        d[len(text[0].text)] = 1
        return d

    def surrounding_tokens(self, text, d):
        """
        Sets the previous and next token as features
        """
        if text[1]:
            d['before'] = text[1].text
        if text[2]:
            d['after'] = text[2].text
        return d

def read_data(filename):
    X_feats = []
    y_feats = []
    with open(filename, encoding='utf-8') as f:

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

def train_SVC(X_train, y_train):
    classifier = svm.SVC(kernel='linear').fit(X_train, y_train)

    return classifier

def train_NB(X_train, y_train):
    classifier = MultinomialNB().fit(X_train, y_train)
    
    return classifier


def main():

    # mode = 'CrossValidation'
    # features = ['first_word', 'surrounding_tokens']

    # Uncomment the above two lines and comment the next five lines of code to make the code work in iPython

    mode = sys.argv[1]
    if sys.argv[2:]:
        features = sys.argv[2:]
    else:
        features = []

    featurizer = Featurizer(features)
    vectorizer = DictVectorizer(sort=False)
    
    if mode == "CrossValidation":
        print("Cross validation may take a while to show its first results.")
        X_feats, y_feats = read_data("gro-ner-train.csv")

        X_feats = np.array(X_feats)
        y_feats = np.array(y_feats)

        prec_total = 0
        rec_total = 0
        f1_total = 0

        for train_index, test_index in KFold(n_splits=10, random_state=None, shuffle=False).split(X_feats):
            X_train, X_test = X_feats[train_index], X_feats[test_index]
            y_train, y_test = y_feats[train_index], y_feats[test_index]

            X_train_dict = featurizer.fit_transform(X_train)
            X_test_dict = featurizer.transform(X_test)

            X_train_ = vectorizer.fit_transform(X_train_dict)
            X_test_ = vectorizer.transform(X_test_dict)

            classifier = train_SVC(X_train_, y_train)
            
            y_pred = classifier.predict(X_test_)

            prec = sklearn.metrics.precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = sklearn.metrics.recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = sklearn.metrics.f1_score(y_test, y_pred, average="macro", zero_division=0)

            prec_total += prec
            rec_total += rec
            f1_total += f1

            print(prec)
            print(rec)
            print(f1)

        print("[Average] Precision: {}".format(prec_total/10))
        print("[Average] Recall: {}".format(rec_total/10))
        print("[Average] F1-score: {}".format(f1_total/10))

    elif mode == "SVC_Test":
        X_train, y_train = read_data("gro-ner-train.csv")
        X_test, y_test = read_data("gro-ner-test.csv")

        X_train_dict = featurizer.fit_transform(X_train)
        X_test_dict = featurizer.transform(X_test)

        X_train_ = vectorizer.fit_transform(X_train_dict)
        X_test_ = vectorizer.transform(X_test_dict)

        # SVC
        classifier = train_SVC(X_train_, y_train)
        # Evaluation
        # sklearn.metrics.plot_confusion_matrix(classifier, X_test_, y_test)
        y_pred = classifier.predict(X_test_)
        print("Support vector classifier")
        print(sklearn.metrics.precision_score(y_test, y_pred, average="macro", zero_division=0))
        print(sklearn.metrics.recall_score(y_test, y_pred, average="macro", zero_division=0))
        print(sklearn.metrics.f1_score(y_test, y_pred, average="macro", zero_division=0))

    elif mode == "NB_Test":
        X_train, y_train = read_data("gro-ner-train.csv")
        X_test, y_test = read_data("gro-ner-test.csv")

        X_train_dict = featurizer.fit_transform(X_train)
        X_test_dict = featurizer.transform(X_test)

        X_train_ = vectorizer.fit_transform(X_train_dict)
        X_test_ = vectorizer.transform(X_test_dict)

        # NB
        classifier = train_NB(X_train_, y_train)

        # Evaluation
        # sklearn.metrics.plot_confusion_matrix(classifier, X_test_, y_test)
        y_pred = classifier.predict(X_test_)
        print("Multinomial NB")
        print(sklearn.metrics.precision_score(y_test, y_pred, average="macro", zero_division=0))
        print(sklearn.metrics.recall_score(y_test, y_pred, average="macro", zero_division=0))
        print(sklearn.metrics.f1_score(y_test, y_pred, average="macro", zero_division=0))
    else:
        sys.stderror.write("Please give one of (CrossValidation | Test_SVC | Test_NB) as a first command line argument\nE.g.: python3 Model_DCC.py CrossValidation token capital")
        exit()

if __name__ == '__main__':
    main()