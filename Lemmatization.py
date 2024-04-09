import nltk
import nltk.corpus
nltk.download("punkt")
nltk.download("wordnet")

#Lemmatization

from nltk.stem import WordNetLemmatizer

lmt = WordNetLemmatizer()
print("\nLemmatization of Words: ")
print("Geese: ", lmt.lemmatize("geese"))
print("Cacti: ", lmt.lemmatize("cacti"))