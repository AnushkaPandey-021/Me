import nltk
import nltk.corpus
nltk.download("punkt")
nltk.download("wordnet")

#Stemming
from nltk.stem import PorterStemmer

pst = PorterStemmer()

print("Stemming of words: \n\n")
print("Buying",pst.stem("Buying"))
print("Going",pst.stem("Going"))
print("Studying",pst.stem("Studying"))
print("Walking",pst.stem("Walking"))
