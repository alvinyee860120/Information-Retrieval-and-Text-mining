
# coding: utf-8

# In[1]:


#import各項步驟所需套件
import os
import numpy as np
import math                              #可以計算log套件
import re                                #remove(過濾的套件)
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize  #tokenize
nltk.download('punkt')
nltk.download('wordnet') 
from nltk.corpus import stopwords        #stopwords
from nltk.stem import PorterStemmer      #stemming
from nltk.stem import WordNetLemmatizer  #lemmatize
from collections import Counter          #資料集count計算


# In[2]:


directory = os.getcwd() #抓取工作目錄
path = os.path.join(directory,'IRTM') #合併工作目錄與目標資料夾形成路徑
alltxt = os.listdir(path) #IRTM資料夾所在完整路徑
df={}
tfidf={}
remove = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  #欲刪除的標點符號
ps = PorterStemmer()
wnl = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) #使用英文文法中的stopwords


# In[3]:


def sortdic(dic): #排序輸入進來的dic型態資料
    sortdic = {}
    for k, v in sorted(dic.items(), key=lambda y: y[0]): #從第一筆資料開始進行排序
        sortdic[k] = v
    return sortdic


# In[4]:


for file in alltxt:
    with open(path+"/"+file) as f:
        #preprocessing steps
        nopunc = re.sub(remove,"", f.read())
        low = nopunc.lower()
        #tokenize = word_tokenize(lower) 此時用wordtokenize做tokenize時，發現會讓單字被切得很奇怪(ex:token->'toke','n')
        tokenize = low.split() #改用split切字
        
        #stem = [ps.stem(w) for w in tokenize]   不做stemming是因為會讓許多字還原成缺少字尾的詞幹，故保留
        lemma = [wnl.lemmatize(t, pos='v') for t in tokenize] #非現在式動詞還原
        lemma = [wnl.lemmatize(t, pos='n') for t in lemma]    #複數名詞還原
        lemma = [wnl.lemmatize(t, pos='a') for t in lemma]    #形容詞還原
        #連做三個不同lemma(詞形還原)的原因是因為相比Porterstem來說我認為更能保留原本字的樣貌
        
        corpus = [w for w in lemma if not w in stop_words] 
        #preprocessing finished
        
        tf = sortdic(Counter(corpus))
        #print(tf)
        
        tfidf[file] = tf
        for term in tf:
            df[term] = df.get(term,0)+1   #紀錄各個term的df


# In[5]:


df = sortdic(df)
#print(df)


# In[6]:


i = 1
t_index={}
for token in df:
    t_index[token] = i
    i += 1
#print(list(df.items()))


# In[7]:


dictionary = pd.DataFrame(list(df.items()), columns=['term', 'df'])
dictionary.index = np.arange(1, len(dictionary) + 1)
dictionary.to_csv('dictionary.csv', index_label='t_index', sep=",")
#print(dictionary)


# In[8]:


idf={}
for term, dfreq in df.items():
    idf[term] = math.log(len(alltxt)/dfreq, 10)  #計算idf值==>以log10為底


# In[9]:


for k, v in tfidf.items():                       #計算每個term的tf*idf值
    for term, tf in v.items():
        v[term] = tf * idf[term]
        #print(v[term])


# In[10]:


def cosine(doc1, doc2):
    x_tfidf = count_tfidf(doc1)
    y_tfidf = count_tfidf(doc2)   
    X = []
    Y = []
    x = x_tfidf.popitem()
    y = y_tfidf.popitem()
    while len(x_tfidf) != 0 and len(y_tfidf) != 0:   
        if x[0] == y[0]:                             
            X.append(x[1])
            Y.append(y[1])
            x = x_tfidf.popitem()
            y = y_tfidf.popitem()
        elif x[0] > y[0]:
            x = x_tfidf.popitem()
        else:
            y = y_tfidf.popitem()
    cos_sim = np.dot(X,Y)            #內積兩不同長度array時會出現error
    return cos_sim
#當兩個array長度不同時，以內積的想法來看的話，較長的dic多出來的vector會對應到另一邊短dic的0向量
#內積後仍然會是0，故遇到長度不同的array時用dic.popitem(),把多的vector移除掉，意義一樣相同


# In[11]:


def output_tfidf(txtname,uv_tfidf):
    f = open(txtname.replace('txt','csv'),'w')
    count=len(uv_tfidf)
    f.write(str(count)+'\n')
    f.close
    output = pd.DataFrame(list(uv_tfidf.items()),columns=['     t_index','tf-idf'])
    output.to_csv(txtname.replace('txt','csv'), index=False, sep=',',mode='a')


# In[12]:


def count_tfidf(txtname):
    txt_tfidf = tfidf[txtname]
    output={}
    vlen = [v for k, v in txt_tfidf.items()]
    vlen = np.linalg.norm(vlen)
    for k,v in txt_tfidf.items():
        output[t_index[k]] = v/(vlen)
    return output  #目標textfile中的各別term的unit vector based的tfidf值 (tfidf值視為該term的vector)


# In[13]:


file1 = '1.txt'
output_tfidf(file1, count_tfidf(file1))
#print(count_tfidf(file1))
print(cosine('1.txt', '2.txt'))

