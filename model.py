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


latent_dim = 256
class create_encoder(keras.Model):
    def __init__(self,encoder_inputs,text_voc_size):
        super().__init__()
        self.encoder_emb = Embedding(text_voc_size, latent_dim,trainable=True,name="encoder_emb")(encoder_inputs)
        self.encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,name="lstm1",dropout=0.1)
        self.encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,name="lstm2",dropout=0.1)
        self.encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True,name="lstm3",dropout=0.1)

    def call(self,training=False):
        encoder_output1, state_h1, state_c1 = self.encoder_lstm1(self.encoder_emb)
        encoder_output2, state_h2, state_c2 = self.encoder_lstm2(encoder_output1)
        encoder_outputs, state_h, state_c = self.encoder_lstm3(encoder_output2)
        return encoder_outputs,state_h,state_c
    

class create_decoder(keras.Model):
    def __init__(self, summary_voc_size, latent_dim):
        super().__init__()
        self.decoder_emb_layer = Embedding(summary_voc_size, latent_dim, trainable=True, name="decoder_emb")
        self.decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm", dropout=0.1)
        self.decoder_dense = TimeDistributed(Dense(summary_voc_size, activation='softmax'))
        
    def call(self, decoder_inputs, context_vector=None):
        decoder_emb = self.decoder_emb_layer(decoder_inputs)
        decoder_outputs, decoder_h_state, decoder_c_state = self.decoder_lstm(decoder_emb, initial_state=[context_vector[0], context_vector[1]])
        decoder_output = self.decoder_dense(decoder_outputs)
        return decoder_output, decoder_h_state, decoder_c_state