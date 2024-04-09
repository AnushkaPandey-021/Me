import nltk
from nltk import ngrams

sentence = 'Sam gave $100 to Max, Isla gave 3000 rupees to Kylie'

unigrams = ngrams(sentence.split(),1)
print("Unigrams:\n")
for grams in unigrams:
    print(grams)
    
bigrams = ngrams(sentence.split(),2)
print("Bigrams:\n")
for grams in bigrams:
    print(grams)
    
trigrams = ngrams(sentence.split(),3)
print("Trigrams:\n")
for grams in trigrams:
    print(grams)
    