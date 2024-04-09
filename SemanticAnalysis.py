import os
import pickle
import random
import re
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import names, stopwords
from nltk.tokenize import word_tokenize
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
nltk.download('names')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

data = pd.read_csv("C:\\IMDB Dataset 1.csv")
dataset = data[:2499]

all_words = []
document = []

stop_words = set(stopwords.words('english'))
all_word_types = ['J']

def preprocess_review(review):
    cleaned = re.sub(r"[^(a-zA-z)\s]", "", review)
    tokenized = word_tokenize(cleaned)
    pos_tags = nltk.pos_tag(tokenized)
    adjectives = [
        w.lower() 
        for w, pos in pos_tags 
        if w.lower() not in stop_words and pos[0].lower()=='j'
    ]
    return adjectives

for index, row in dataset.iterrows():
    adjectives = preprocess_review(row['review'])
    document.append((adjectives,row['sentiment']))
    all_words.extend(adjectives)
    
all_words = nltk.FreqDist(all_words)
print("Frequency Distribution of all the words:", all_words)

if all_words:
    all_words.plot(30,cumulative=False)
    plt.show()
    
word_features = list(all_words.keys())

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = w in words
        return features if features else {'contains_no_features':True}

featuresets = [(find_features(rev), category) for(rev, category) in document]

random.shuffle(featuresets)

train_set, test_set = train_test_split(featuresets, test_size=0.2, random_state=42)

classifier = nltk.NaiveBayesClassifier.train(train_set)
print("Naive Bayes Classifier accuracy", (nltk.classify.accuracy(classifier, test_set))*100)

MNB_clf = SklearnClassifier(MultinomialNB())
mnb_cls = MNB_clf.train(train_set)
print("Multinomial NB Classifier accuracy ", (nltk.classify.accuracy(mnb_cls,test_set))*100)

BNB_clf = SklearnClassifier(BernoulliNB())
bnb_cls = BNB_clf.train(train_set)
print("Bernoulli NB Classifier accuracy ", (nltk.classify.accuracy(bnb_cls,test_set))*100)




