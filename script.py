import pandas as pd
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Concatenate, LSTM, Embedding, Input, Bidirectional, Dense, RepeatVector, Concatenate, Activation, Dot, Lambda
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
    

data['text_clean'] = data['text'].apply(lambda x:text_cleaner(x))
data['summary_clean'] = data['summary'].apply(lambda x:summary_cleaner(x))
data['summary_clean_input'] = data['summary_clean'].apply(lambda x : '_START_ ' + x )
data['summary_clean'] = data['summary_clean'].apply(lambda x : x + ' _END_')



data['text_len'] = data['text_clean'].str.split().str.len()
data['summary_len'] = data['summary_clean'].str.split().str.len()

length_df = data[['text_len','summary_len']]
length_df.hist(bins = 30)
#data['summary_len'].value_counts()

max_len_input = 90
max_len_output = 10

data = data[(data['text_len']<=90)&(data['summary_len']<=10)]

filtered = data[:5000]
 
        
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


tokenizer_inputs = Tokenizer()
tokenizer_inputs.fit_on_texts(list(filtered['text_clean']))
input_sequences = tokenizer_inputs.texts_to_sequences(filtered['text_clean'])
word2idx_inputs = tokenizer_inputs.word_index
num_words_input = len(tokenizer_inputs.word_index)+1

tokenizer_outputs = Tokenizer(filters = '')
tokenizer_outputs.fit_on_texts(list(filtered['summary_clean'])+list(filtered['summary_clean_input']))
target_sequences = tokenizer_outputs.texts_to_sequences(filtered['summary_clean'])
target_sequences_input = tokenizer_outputs.texts_to_sequences(filtered['summary_clean_input'])
word2idx_outputs = tokenizer_outputs.word_index
num_words_output = len(tokenizer_outputs.word_index)+1

encoder_inputs = pad_sequences(input_sequences, maxlen = max_len_input )
decoder_inputs = pad_sequences(target_sequences_input, maxlen = max_len_output, padding = 'post')
decoder_targets = pad_sequences(target_sequences, maxlen = max_len_output, padding = 'post')


input_embed = embeddings()
input_embedding_weights = input_embed.get_embedding_matrix(num_words_input,tokenizer_inputs)

output_embed = embeddings()
output_embedding_weights = output_embed.get_embedding_matrix(num_words_output,tokenizer_outputs)


encoder_embedding = Embedding(num_words_input,50,weights = [input_embedding_weights],input_length = max_len_input)

decoder_embedding = Embedding(num_words_output,50,weights = [output_embedding_weights],input_length = max_len_output)


decoder_targets_one_hot = np.zeros((filtered.shape[0], max_len_output, num_words_output),dtype='float32')

for i, d in enumerate(decoder_targets):
  for t, word in enumerate(d):
    if word != 0:
      decoder_targets_one_hot[i, t, word] = 1
      
def softmax_over_time(x):
  assert(K.ndim(x) > 2)
  e = K.exp(x - K.max(x, axis=1, keepdims=True))
  s = K.sum(e, axis=1, keepdims=True)
  return e / s

attn_repeat = RepeatVector(max_len_input)
attn_concat = Concatenate(axis=-1)
attn_dense1 = Dense(10,activation='tanh')
attn_dense2 = Dense(1, activation = softmax_over_time)
attn_dot = Dot(axes=1)


def one_step_attention(h,st_1):
    


