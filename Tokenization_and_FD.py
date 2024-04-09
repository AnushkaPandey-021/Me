import nltk
import nltk.corpus
nltk.download("punkt")
nltk.download("wordnet")

#Tokenization
from nltk.tokenize import word_tokenize
sample="There will always be people who say mean words because you are different. And sometimes, their minds cannot be changed. But there are many more people that do not judge people based on how they look, or where they came from. Those are the people whose words truly matter."
Sample_Tokens = word_tokenize(sample)
print("Sample Tokens: \n",Sample_Tokens)
print("Type of Sample Tokens: \n",type(Sample_Tokens))
print("Lenght of Sample Tokens: \n",len(Sample_Tokens))

#Frequency Distribution
from nltk.probability import FreqDist
FDist = FreqDist(Sample_Tokens)
print("Frequency Distribution of Sample Tokens: \n", FDist)
top5 = FDist.most_common(5)
print("Top 5 words: \n", top5)