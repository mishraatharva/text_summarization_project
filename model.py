from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
# from constants import *


from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam
import keras


latent_dim = 500 

latent_dim = 500 
class create_encoder(keras.Model):
    def __init__(self,x_voc_size):
        super().__init__()
        self.encoder_emb = Embedding(x_voc_size, latent_dim,trainable=True,name="encoder_emb")
        self.encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,name="lstm1")
        self.dropout_one = Dropout((0.4),name="dropout_one")
        self.encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,name="lstm2")
        self.dropout_two = Dropout((0.2),name="dropout_two")
        self.encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True,name="lstm3")

    def call(self,inputs,*args, **kwargs):
        x = self.encoder_emb(inputs)
        encoder_output1, state_h1, state_c1 = self.encoder_lstm1(x)
        x = self.dropout_one(encoder_output1)
        encoder_output2, state_h2, state_c2 = self.encoder_lstm2(x)
        x = self.dropout_two(encoder_output2)
        encoder_outputs, state_h, state_c = self.encoder_lstm3(x)
        return encoder_outputs,state_h,state_c



# class create_encoder(keras.Model):
#     def __init__(self,x_voc_size):
#         super().__init__()
#         self.encoder_emb = Embedding(x_voc_size, latent_dim,trainable=True,name="encoder_emb")
#         self.encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,name="lstm1")
#         self.dropout_one = Dropout((0.4),name="dropout_one")
#         self.encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,name="lstm2")
#         self.dropout_two = Dropout((0.2),name="dropout_two")
#         self.encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True,name="lstm3")

#     def call(self,inputs,*args, **kwargs):
#         x = self.encoder_emb(inputs)
#         encoder_output1, state_h1, state_c1 = self.encoder_lstm1(x)
#         x = self.dropout_one(encoder_output1)
#         encoder_output2, state_h2, state_c2 = self.encoder_lstm2(x)
#         x = self.dropout_two(encoder_output2)
#         encoder_outputs, state_h, state_c = self.encoder_lstm3(x)
#         return encoder_outputs,state_h,state_c

class create_decoder(keras.Model):
    def __init__(self,y_voc_size):
        super().__init__()
        self.decoder_emb = Embedding(y_voc_size, latent_dim,trainable=True,name="decoder_emb")
        self.decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,name="decoder_lstm") 
        self.dropout_decoder = Dropout((0.3),name="dropout_decoder")
        self.decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax'))
        
    def call(self, decoder_inputs, context_vector=None):
        decoder_emb = self.decoder_emb(decoder_inputs)
        decoder_outputs,decoder_h_state, decoder_c_state = self.decoder_lstm(decoder_emb,initial_state=[context_vector[0], context_vector[1]])
        x = self.dropout_decoder(decoder_outputs)
        decoder_output = self.decoder_dense(x)
        return decoder_output,decoder_h_state, decoder_c_state


    
# class create_decoder(keras.Model):
#     def __init__(self,y_voc_size):
#         super().__init__()
#         self.decoder_emb = Embedding(y_voc_size, latent_dim,trainable=True,name="decoder_emb")
#         self.decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,name="decoder_lstm") 
#         self.dropout_decoder = Dropout((0.3),name="dropout_decoder")
#         self.decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax'))
        
#     def call(self, decoder_inputs, context_vector=None):
#         decoder_emb = self.decoder_emb(decoder_inputs)
#         decoder_outputs,decoder_h_state, decoder_c_state = self.decoder_lstm(decoder_emb,initial_state=[context_vector[0], context_vector[1]])
#         x = self.dropout_decoder(decoder_outputs)
#         decoder_output = self.decoder_dense(x)
#         return decoder_output,decoder_h_state, decoder_c_state