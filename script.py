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


K.clear_session()

latent_dim = 300
embedding_dim=50

# Encoder
encoder_inputs = Input(shape=(max_len_text,))

#embedding layer
enc_emb =  Embedding(x_vocab, embedding_dim,weights = [x_emb_weights],trainable=False)(encoder_inputs)

#encoder lstm 1
encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

#encoder lstm 2
encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

#encoder lstm 3
encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))

#embedding layer
dec_emb_layer = Embedding(y_vocab, embedding_dim,weights = [y_emb_weights],trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

# Attention layer
attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

# Concat attention input and decoder LSTM output
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

#dense layer
decoder_dense =  TimeDistributed(Dense(y_vocab, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)

# Define the model 
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)

history=model.fit([x_train,y_train[:,:-1]], y_train.reshape(y_train.shape[0],y_train.shape[1], 1)[:,1:] ,epochs=50,callbacks=[es],batch_size=128, validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))


def beam_search_decoder(data, k):
	sequences = [[list(), 1.0]]
	for row in data:
		all_candidates = list()
		for i in range(len(sequences)):
			seq, score = sequences[i]
			for j in range(len(row)):
				candidate = [seq + [j], score * -np.log(row[j])]
				all_candidates.append(candidate)
		ordered = sorted(all_candidates, key=lambda tup:tup[1])
		sequences = ordered[:k]
	return sequences

def sample_line():
  # initial inputs
  np_input = np.array([[ word2idx['<sos>'] ]])
  h = np.zeros((1, LATENT_DIM))
  c = np.zeros((1, LATENT_DIM))

  # so we know when to quit
  eos = word2idx['<eos>']

  # store the output here
  output_sentence = []

  for _ in range(max_sequence_length):
    o, h, c = sampling_model.predict([np_input, h, c])

    # print("o.shape:", o.shape, o[0,0,:10])
    # idx = np.argmax(o[0,0])
    probs = o[0,0]
    if np.argmax(probs) == 0:
      print("wtf")
    probs[0] = 0
    probs /= probs.sum()
    idx = np.random.choice(len(probs), p=probs)
    if idx == eos:
      break

    # accuulate output
    output_sentence.append(idx2word.get(idx, '<WTF %s>' % idx))

    # make the next input into model
    np_input[0,0] = idx

  return ' '.join(output_sentence)