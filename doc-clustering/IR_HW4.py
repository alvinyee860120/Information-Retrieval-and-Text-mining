
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


def preprocessing(file):
    #remove 標點符號
    nopunc = re.sub(remove,"", file)
    #字轉小寫
    low = nopunc.lower()
    #tokenize = word_tokenize(lower) 此時用wordtokenize做tokenize時，發現會讓單字被切得很奇怪(ex:token->'toke','n')
    tokenize = low.split() #改用split切字
    
    #stemming
#     stem = [ps.stem(w) for w in tokenize]   #不做stemming是因為會讓許多字還原成缺少字尾的詞幹，故保留==>效果做出來比lemmatize差
    
    #連做三個不同lemma(詞形還原)的原因是因為相比Porterstem來說我認為更能保留原本字的樣貌
    lemma = [wnl.lemmatize(t, pos='v') for t in tokenize] #非現在式動詞還原
    lemma = [wnl.lemmatize(t, pos='n') for t in lemma]    #複數名詞還原
    lemma = [wnl.lemmatize(t, pos='a') for t in lemma]    #形容詞還原
    
    #remove stopwords
    corpus = [w for w in lemma if not w in stop_words]
    return corpus


# In[5]:


for file in alltxt:
    with open(path+"/"+file) as f:
        #preprocessing steps
        corpus = preprocessing(f.read())
        
        #compute each file's tf value
        tf = sortdic(Counter(corpus))
        tfidf[file] = tf
        #compute df value
        for term in tf:
            df[term] = df.get(term,0)+1   #紀錄各個term的df值


# In[6]:


df = sortdic(df)
#print(df)


# In[7]:


idf={}
for term, dfreq in df.items():
    idf[term] = math.log(len(alltxt)/dfreq, 10)  #計算idf值==>以log10為底


# In[8]:


for k, v in tfidf.items():                       #計算每個term的tf*idf值
    for term, tf in v.items():
        v[term] = tf * idf[term]


# In[9]:


def cosine(doc1, doc2):
    X = all_tfidf[doc1]
    Y = all_tfidf[doc2]
    cos_sim = np.dot(X,Y)            
    return cos_sim


# In[10]:


def count_txt_tfidf(txtname):
    txt_tfidf = tfidf[txtname]
    BOW = []
    for term, dfreq in df.items():
        if term in txt_tfidf:
            BOW.append(txt_tfidf[term])
        elif term not in txt_tfidf:
            BOW.append(0)
    
    vlen = np.linalg.norm(BOW)
    for i in range(len(BOW)):
        BOW[i] = BOW[i]/(vlen)
    BOW = np.array(BOW)
    return BOW  #目標textfile中的各別term的unit vector based的 tfidf 值 (tfidf值視為該term的vector)


# In[11]:


all_tfidf = {}
for i in range(1095):
    all_tfidf[i+1] = (count_txt_tfidf(str(i+1)+'.txt'))
    print(i+1)


# In[12]:


a = np.zeros((1095,1095)) #1095*1095維的 cosine similarity
cosine_list = []
for n in range(1095):
    print(n)
    for i in range(1095):
        print(i)
        if n == i:        
            a[n][i] = -1 #自己跟自己的文本cos_sim必定為1.0，在此設為-1，方便之後操作
        else:
            cos_sim = cosine((n+1),(i+1))
            a[n][i] = cos_sim
print(a) #pairwise_cos_sim


# In[13]:


np.save('cos_sim', a) #輸出cosine similarity table，之後只需載入就可以讀取資料
# pairwise_cos_sim = np.load('cos_sim.npy')


# In[14]:


def HAC_clustering(a2,ic):
    A = [] #merge list
    I = np.ones(1095)
    for k in range(1095-ic):
        print(k)
        m = np.argmax(a2) #找最大值所在的index
        r, c = divmod(m, a2.shape[1]) #找index所在的row及column位置
        if I[r]!=0 and I[c]!=0:
            A.append([r,c])   
            small = min(r,c)
            large = max(r,c)
            I[large] = 0 # update I
            for i in range(1095): # update table
                if a2[small][i] < a2[large][i]: #complete_link
                    a2[large][i] = -1 
                    a2[i][large] = -1
                else:
                    a2[small][i] = a2[large][i]
                    a2[i][small] = a2[i][large] 
                    a2[large][i] = -1 
                    a2[i][large] = -1            
    print(len(A))
    return A


# In[15]:


pairwise_cos_sim = np.load('cos_sim.npy')
eight_cluster = HAC_clustering(pairwise_cos_sim,8)
# print(len(eight_cluster))
# eight_cluster


# In[16]:


pairwise_cos_sim = np.load('cos_sim.npy')
thirteen_cluster = HAC_clustering(pairwise_cos_sim,13)
# print(len(thirteen_cluster))
# thirteen_cluster


# In[17]:


pairwise_cos_sim = np.load('cos_sim.npy')
twenty_cluster = HAC_clustering(pairwise_cos_sim,20)
# print(len(twenty_cluster))
# twenty_cluster


# In[18]:


def construct_tree(clustering):
    k = []
    for i in range(1095):
        k.append(0)
    for i in range(len(clustering)):
#         print('--------',i)
        if k[clustering[i][0]] == 0 and k[clustering[i][1]] == 0:
#             print('a')
            k[clustering[i][0]] = clustering[i]
#             print(k[clustering[i][0]])
        elif k[clustering[i][0]] == 0 and k[clustering[i][1]] != 0:
#             print('b')
            k[clustering[i][1]].extend(clustering[i])
            k[clustering[i][0]] = k[clustering[i][1]]
            k[clustering[i][0]] = list(np.unique(k[clustering[i][0]]))
            k[clustering[i][1]] = 0
#             print(k[clustering[i][0]])
#             print(k[clustering[i][1]])
        elif k[clustering[i][0]] != 0 and k[clustering[i][1]] == 0:
#             print('c')
            k[clustering[i][0]].extend(clustering[i])
            k[clustering[i][0]] = list(np.unique(k[clustering[i][0]]))
#             print(k[clustering[i][0]])
        else:
#             print('d')
            k[clustering[i][0]].extend(k[clustering[i][1]])
            k[clustering[i][1]] = 0
#             print(k[clustering[i][0]])
#             print(k[clustering[i][1]])
    return k 


# In[19]:


eight_tree = construct_tree(eight_cluster)


# In[20]:


thirteen_tree = construct_tree(thirteen_cluster)


# In[21]:


twenty_tree = construct_tree(twenty_cluster)


# In[22]:


def check(k):
    for i in range(len(k)):
        if k[i] != 0:
            print(len(k[i]))
            #print(k[i])
            print('\n')


# In[23]:


print('--------eight-------')
check(eight_tree)
print('--------thirteen-------')
check(thirteen_tree)
print('--------twenty-------')
check(twenty_tree)


# In[24]:


def output_textfile(n, k):
    with open(str(n)+".txt", "w") as f:
        for i in range(len(k)):
            if k[i] != 0:
                length = len(k[i])
                print(length)
                for ic in range(length):
                    k[i] = sorted(k[i])
                    f.write(str(k[i][ic]+1)+'\n')
                f.write('\n')


# In[25]:


print('--------eight-------')
output_textfile(8, eight_tree)
print('--------thirteen-------')
output_textfile(13, thirteen_tree)
print('--------twenty-------')
output_textfile(20, twenty_tree)

