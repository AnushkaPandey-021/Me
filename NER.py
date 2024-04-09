import nltk
from nltk.tokenize import word_tokenize
from nltk import ne_chunk

# Download necessary resources
nltk.download('maxent_ne_chunker')
nltk.download('wordnet')
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Sample text
sample = 'My name is Anushka Pandey.I work for Microsoft org. I like to listen to music.' \
         'The quick brown fox jumps over the lazy dog.' \
         'Should I audition for the show?' \
         'How dare they!'

# Tokenize the text
sample_tokens = word_tokenize(sample)

# Perform NER on the sample text
sample_tag = nltk.pos_tag(sample_tokens)
sample_ner = ne_chunk(sample_tag)
print(sample_ner)
