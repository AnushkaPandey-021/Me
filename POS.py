import nltk
from nltk import ne_chunk
nltk.download('maxnet_ne_chunker')
nltk.download('wornet')
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

sample = "There will always be people who say mean words because you are different. And sometimes, their minds cannot be changed. But there are many more people that do not judge people based on how they look, or where they came from. Those are the people whose words truly matter."

sample_tokens = word_tokenize(sample)

for i in sample_tokens:
    print(nltk.pos_tag([i]))

