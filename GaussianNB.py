import pandas as pd
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

dataset =[["I liked the movie", "positive"],
["It's a good movie. Nice Story","positive"],
["Hero's acting is bad but heroine looks good. Overall nice movie","positive"],
["Nice songs. But sadly boring ending","negative"],
["Sad movie, boring movie","negative"]]

dataset = pd.DataFrame(dataset)
dataset.columns = ['Text','Reviews']
corpus =[]

for i in range(0,5):
    text = re.sub('[^a-zA-Z]',' ',dataset['Text'][i])
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if word not in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpus.append(text)
print(corpus)

cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=0)

classifier=GaussianNB()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
    
