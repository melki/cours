
# coding: utf-8

# In[83]:

import numpy as  np
import codecs
import matplotlib.pyplot as plt
import unicodedata
import re
from collections import Counter,defaultdict
import nltk.corpus.reader as pt
import os

import string
import pdb
import nltk
import scipy
import glob, os
import sklearn


# In[89]:

def readAFile(nf):
    f = open(nf, 'rb')
    l = []
    txt = f.readlines()
    for i in txt:
        l.append(i.decode("utf-8"))
    f.close()
    l = ' '.join(l)
    return l


# In[90]:

def process(txt):
    #txt = txt[txt.find("\n\n"):] # elimination de l'entete (on ne conserve que les caractères après la première occurence du motif
    txt = unicodedata.normalize("NFKD",txt).encode("ascii","ignore") # elimination des caractères spéciaux, accents...

    punc = string.punctuation    # recupération de la ponctuation
    punc += u'\n\r\t\\'          # ajouts de caractères à enlever
    table =string.maketrans(punc, ' '*len(punc))  # table de conversion punc -> espace
    txt = string.translate(txt,table).lower() # elimination des accents + minuscules
    return txt
    #return re.sub(" +"," ", txt) # expression régulière pour transformer les espaces multiples en simples espaces


# In[91]:

#read positif 
os.chdir("/home/melki/Documents/cours/tal/tme8/pos")
pos = []
for file in glob.glob("*.txt"):
    pos.append(readAFile(file))
print len(pos)

#read positif 
os.chdir("/home/melki/Documents/cours/tal/tme8/neg")
neg = []
for file in glob.glob("*.txt"):
    neg.append(readAFile(file))
print len(neg)


# In[ ]:




# In[92]:

X = pos + neg
Y = [1 for i in range(len(pos))]+[-1 for i in range(len(neg))]
print len(X)
print len(Y)


# In[93]:

from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer

stopList = []

# On instancie le tokenizer
tokenizer = WordPunctTokenizer()
french_stopwords = set(stopwords.words('english'))

for a in range(len(X)):

    # L'interface étant identique, le reste du code est le même
    tokens = tokenizer.tokenize(X[a])


    # chargement des stopwords français

    # un petit filtre
    tokens = [token for token in tokens if token.lower() not in french_stopwords]

    stopList.append(' '.join(tokens))



# In[97]:

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation



countVe = CountVectorizer(strip_accents='unicode',max_df=0.95, min_df=1, analyzer="word",ngram_range=(1, 3) )

tfve = TfidfVectorizer(strip_accents='unicode' ,max_df=0.9, min_df=1, analyzer="word",ngram_range=(1, 3) )


# In[98]:

idf = tfve.fit_transform(stopList)


# In[99]:

idf


# In[100]:

import numpy as np
import sklearn.naive_bayes as nb
from sklearn import svm
from sklearn import linear_model as lin

# données ultra basiques, à remplacer par vos corpus vectorisés
X = idf
# X = count
y = np.array(Y)

# SVM
clf = svm.LinearSVC(C=1,max_iter=1000000)
# apprentissage
clf.fit(X, y)  


# In[101]:

os.chdir("/home/melki/Documents/cours/tal/tme8")
testA = readAFile("testSentiment.txt")

test=[process(unicode(xi)) for xi in test]

testList = []

# On instancie le tokenizer
tokenizer = WordPunctTokenizer()
french_stopwords = set(stopwords.words('english'))

for a in range(len(test)):

    # L'interface étant identique, le reste du code est le même
    tokens = tokenizer.tokenize(test[a])


    # chargement des stopwords français

    # un petit filtre
    tokens = [token for token in tokens if token.lower() not in french_stopwords]

    testList.append(' '.join(tokens))


# In[ ]:




# In[102]:

a = tfve.transform(test)

pp = clf.predict(a) 


# In[103]:

print(test[0])


# In[104]:


f = open('stm.txt', 'w')
for i in pp:
    if i == -1:
        f.write('C\n')
    else:
        f.write('M\n')
f.close()


# In[105]:

len(pp)


# In[31]:




# In[ ]:



