
# coding: utf-8

# In[7]:


#import各項步驟所需套件
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# In[8]:


#Oringinal text(one english doc from https://ceiba.ntu.edu.tw/course/35d27d/content/28.txt )
text = "And Yugoslav authorities are planning the arrest of eleven coal miners and two opposition politicians on suspicion of sabotage, that's in connection with strike action against President Slobodan Milosevic. You are listening to BBC news for The World."
print(text)
print('\n')


# In[9]:


#Step1: lowercase everything in the text.
text = text.lower()
print(text)
print('\n')


# In[10]:


#Step2: Tokenization (split the text into each word token). 
word = word_tokenize(text)
print(word)
print('\n')


# In[12]:


#Step3: Stemming using Porter's algo (reduces related words to a common stem).
ps = PorterStemmer()
stem = [ps.stem(w) for w in word]
print (stem)
print('\n')
#for w in word:
    #print(ps.stem(w))


# In[13]:


#Step4: Stopwords Removal (removing stopwords such as‘a’, ‘the’, ‘is’, which is unlikelyto be benefit in NLP)
stop_words = set(stopwords.words('english'))
#print(stop_words)
#showing stopwords actually look like.


# In[14]:


#remove stopwords
meaningful_words = [w for w in stem if not w in stop_words]
print(meaningful_words)


# In[15]:


f = open("result.txt", "w") #open(創建)一個new txt file called 'result', "w"(Write):Opens a file for writing, creates the file if it does not exist
f.write(str(meaningful_words)) #利用write fuction 將 meaningful_words寫進file中
f.write('\n')
f.close()

