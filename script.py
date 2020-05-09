import pandas as pd
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from attention_keras.layers.attention import AttentionLayer
from tensorflow.python.keras.layers import Concatenate, LSTM, Embedding, Input, TimeDistributed, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping



data = pd.read_csv('Reviews.csv')
columns = list(data.columns)
new_columns = []
for col in columns:
    new_columns.append(col.lower())
    
data.columns = new_columns

data = data.drop_duplicates(subset='text')
data = data.dropna()

check = data['text'][20]

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}

stop_words = set(stopwords.words('english'))
def text_cleaner(x):
    x = x.lower()
    x = BeautifulSoup(x, "lxml").text
    x = ' '.join([ contraction_mapping[t] if t in contraction_mapping else t for t in x.split()])
    x = re.sub(r"'s\b","",x)
    x = re.sub('[^a-zA-Z]',' ',x)
    tokens = [w for w in x.split() if not w in stop_words]
    long_words=[]
    for i in tokens:
        if len(i)>=3:
            long_words.append(i)   
    return (" ".join(long_words)).strip()



def summary_cleaner(x):
    x = re.sub('"','', x)
    x = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in x.split(" ")])    
    x = re.sub(r"'s\b","",x)
    x = re.sub("[^a-zA-Z]", " ", x)
    x = x.lower()
    tokens=x.split()
    long_words = [] 
    for i in tokens:
        if len(i)>1:  
            long_words.append(i)                               
    return (" ".join(long_words)).strip()
    

data['new_text'] = data['text'].apply(lambda x:text_cleaner(x))
data['new_summary'] = data['summary'].apply(lambda x:summary_cleaner(x))
data['new_summary'] = data['new_summary'].apply(lambda x : '_START_ '+ x + ' _END_')


data['text_len'] = data['new_text'].str.split().str.len()
data['summary_len'] = data['new_summary'].str.split().str.len()

length_df = data[['text_len','summary_len']]
length_df.hist(bins = 30)

max_len_text=90
max_len_summary=10

data = data[(data['text_len']<=90)&(data['summary_len']<=10)]

filtered = data[:100000]
 
        
class embeddings():
    def __init__(self):
        self._embeddings_index = dict()
        with open(r'glove\glove.6B.50d.txt','r',encoding='utf-8') as f:
            for line in f:
            	values = line.split()
            	word = values[0]
            	coefs = np.array(values[1:], dtype=np.float64)
            	self._embeddings_index[word] = coefs
            f.close()
    
    def get_embedding_matrix(self,vocab_size,tokenizer):
        ''' Embdedding matrix for words used in the input'''
        embedding_matrix = np.zeros((vocab_size, 50))
        for word, i in tokenizer.word_index.items():
        	embedding_vector = self._embeddings_index.get(word)
        	if embedding_vector is not None:
        		embedding_matrix[i] = embedding_vector
                
        return embedding_matrix

x_train,x_val,y_train,y_val=train_test_split(np.array(filtered['text']),np.array(filtered['summary']),test_size=0.1,random_state=0,shuffle=True)


x_tokenizer = Tokenizer() 
x_tokenizer.fit_on_texts(list(x_train))
x_tr_seq    =   x_tokenizer.texts_to_sequences(x_train) 
x_val_seq   =   x_tokenizer.texts_to_sequences(x_val)
x_train    =   pad_sequences(x_tr_seq,  maxlen=max_len_text, padding='post')
x_val   =   pad_sequences(x_val_seq, maxlen=max_len_text, padding='post')
x_vocab   =  len(x_tokenizer.word_index) + 1

y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(list(y_train))
y_tr_seq    =   y_tokenizer.texts_to_sequences(y_train) 
y_val_seq   =   y_tokenizer.texts_to_sequences(y_val) 
y_train    =   pad_sequences(y_tr_seq, maxlen=max_len_summary, padding='post')
y_val   =   pad_sequences(y_val_seq, maxlen=max_len_summary, padding='post')
y_vocab  =   len(y_tokenizer.word_index) +1

e = embeddings()
x_emb_weights = e.get_embedding_matrix(x_vocab,x_tokenizer)
y_emb_weights = e.get_embedding_matrix(y_vocab,y_tokenizer)


