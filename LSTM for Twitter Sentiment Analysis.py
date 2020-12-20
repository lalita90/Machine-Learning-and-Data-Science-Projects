# -*- coding: utf-8 -*-
import keras
import tensorflow as tf
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, GRU
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import Word2Vec
from google.colab import drive
drive.mount("/content/drive")

!ls 'drive/Dataset/Sentiment140'

root = "drive/Dataset/Sentiment140"

#!echo "tmpfs /dev/shm tmpfs defaults,size=100g 0 0" >> ../../../etc/fstab

#!mount -o remount /dev/shm

cols = ['sentiment','id','date','query_string','user','text']
df = pd.read_csv(root + '/train.csv', engine='python',encoding = "ISO-8859-1",names=cols)
# Dataset is now stored in a Pandas Dataframe\

#reading test data
cols = ['sentiment','id','date','query_string','user','text']
test = pd.read_csv(root + '/test.csv', engine='python',encoding = "ISO-8859-1",names=cols)

#reading dev data
cols = ['sentiment','id','date','query_string','user','text']
dev = pd.read_csv(root + '/dev.csv', engine='python',encoding = "ISO-8859-1",names=cols)



# Story Generation and Visualization from Tweets

# A) Understanding the common words used in the tweets: WordCloud

# visualize all the words our data 

all_words = ' '.join([text for text in df.text.apply(str)])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)



# B) Words in racist/sexist tweets
#Expect to see negative words

normal_words =' '.join([text for text in df.text.apply(str)[df['sentiment'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)

# Words in non racist/sexist tweets    
#Expect to see positive words

normal_words =' '.join([text for text in df.text.apply(str)[df['sentiment'] == 4]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)

#load lem data
lem_data = pd.read_pickle(root + '/lem_train.pkl')
lem_dev = pd.read_pickle(root + '/lem_dev.pkl')
lem_test = pd.read_pickle(root + '/lem_test.pkl')

#word2vec

model_ted = Word2Vec(sentences=tok_text, size=10000, window=5, min_count=5, workers=4, sg=0)

#checking similar words
model_ted.wv.most_similar('singer')

tok_text.shape

#checking similar words
model_ted.wv.most_similar('facebook')

Word2Vec(sentences=tok_text, size=1000, window=5, min_count=5, workers=4, sg=0)


### Create sequence

vocabulary_size = 10000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(lem_data)##
sequences = tokenizer.texts_to_sequences(lem_data)
data = pad_sequences(sequences, maxlen=30)          


#tokenizing dev
seq_dev=tokenizer.texts_to_sequences(lem_dev)##
data_dev = pad_sequences(seq_dev,maxlen=30)


# data.shape
df.loc[df.sentiment == 4, 'sentiment'] = 1
test.loc[test.sentiment == 4, 'sentiment'] = 1
test.drop(test[test.sentiment == 2].index, inplace=True)
dev.loc[dev.sentiment == 4, 'sentiment'] = 1

# dev['sentiment']

## Network architecture model 1
model = Sequential()
model.add(Embedding(20000, 100, input_length=30))
model.add(LSTM(10,dropout=0.2))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))                     
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])   



## Network architecture model 2
model = Sequential()
model.add(Embedding(20000, 100, input_length=30))
model.add(LSTM(5, dropout=0.2, recurrent_dropout=0.2))   
model.add(Dense(1, activation='sigmoid'))                    
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  

#model 3

model = Sequential()
model.add(Embedding(20000, 100, input_length=30))
model.add(GRU(5, dropout=0.2, recurrent_dropout=0.2)) 
model.add(Dense(1, activation='sigmoid'))                     
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  

#model4
embedding_size=32
model=Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=30))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])
print(model.summary())



checkpoint_path =root+"/training_1/cp.ckpt" 
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

                 steps_per_epoch=data.shape[0]/128)

type(df['sentiment'])


#model.fit(data, np.array(df['sentiment']), epochs=8,batch_size=128, validation_data=(data_dev, np.array(dev['sentiment'])))
history=model.fit(data, np.array(df['sentiment']),  
          epochs = 10, batch_size=1024,
          validation_data = (data_dev, np.array(dev['sentiment'])),
          callbacks = [cp_callback], shuffle=True)  # pass callback to training

history.history.keys()

"""Plots"""


plt.plot(history.history['acc'], 'b')
plt.title('Training Accuracy vs epoch')


plt.plot(history.history['loss'], 'r')
plt.title('Training Loss vs epoch')

plt.plot(history.history['val_acc'], 'b')
plt.title('Validation Accuracy vs epoch')


plt.plot(history.history['val_loss'], 'r')
plt.title('Validation Loss vs epoch')

my_tags = ['class0','class1']
plt.figure(figsize=(6,4))
df.sentiment.value_counts().plot(kind='bar')


#save model output
!pip install h5py pyyaml 
# Save entire model to a HDF5 file
model.save('model2.h5')
# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('model2.h5')
new_model.summary()


#tokenizing test
seq_test=tokenizer.texts_to_sequences(lem_test)
data_test = pad_sequences(seq_test,maxlen=30)

model.evaluate(x=data_test, y=test['sentiment'])

#download model
#from google import files
from google.colab import files
files.download("model2.h5")
#files.download("model2.h5")

#hourly tweet analysis

#upload model
from google.colab import files
files.upload("model2.h5")


