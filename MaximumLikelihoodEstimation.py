import nltk
from nltk import FreqDist
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize

nltk.download('reuters')
nltk.download('punkt')

corpus = reuters.raw()

tokens = word_tokenize(corpus)

word_freq = FreqDist(tokens)
print("Frequency Distribution of words: \n", word_freq)

total_words = len(tokens)
print("Total words in corpus: \n",total_words)

mle_probability = {word:count / total_words for word, count in word_freq.items()}

example_word = 'economy'
print(f"MLE of '{example_word}' : {mle_probability.get(example_word,0)}")