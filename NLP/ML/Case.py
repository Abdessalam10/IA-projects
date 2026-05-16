import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../../datasets/Restaurant_Reviews.tsv', sep='\t')
#Exploring the dataset
data.info()
print(data['Liked'].value_counts())
sns.countplot(x='Liked', data=data)
#plt.show()
data['Review_length'] = data['Review'].apply(len)
print(data)
print(data.loc[data['Review_length'].idxmax()]['Review'])

#Data Preprocessing
import nltk 
from nltk.corpus import stopwords
nltk.download('stopwords')
#print(stopwords.words('english'))
s=data['Review'][0]
import re
s=re.sub('[^a-zA-Z]', ' ', s)
s=s.lower()
s=s.split()
print(s)
temp = []
for word in s:
    if word not in stopwords.words('english'):
        temp.append(word)
print(temp)

s = ' '.join(temp)

from nltk.stem import  PorterStemmer
ps = PorterStemmer()
s = ps.stem(s)
print(s)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
cv.fit_transform(s.split()).toarray()

corpus = []
for i in range(len(data)):
    s = re.sub('[^a-zA-Z]', ' ', data['Review'][i])
    s = s.lower()
    s = s.split()
    s = [word for word in s if word not in stopwords.words('english')]
    s = ' '.join(s)
    s = ps.stem(s)
    corpus.append(s)
    
print(corpus)
cv=CountVectorizer()
x = cv.fit_transform(corpus).toarray()
print(x.shape)
y=data['Liked'].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)    
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

#Model Deployment

import joblib
joblib.dump(model, 'restaurant_review_model.pkl')
