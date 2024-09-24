from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
import numpy as np
import pickle
import os
from model import create_decoder,create_encoder
from constants import *
from model import *

# trained_model = load_model(r'U:\nlp_project\text_summarization\artifacts\model\ts_model.pickle')

class Create_Encoder_Decoder():
    def __init__(self):
        self.encoder_inputs = Input(MAX_LEN_TEXT)
        self.decoder_inputs = Input(shape=(None,))
        self.x_voc_size = 27979
        self.y_voc_size = 7765
        self.trained_model = load_model(r'U:\nlp_project\text_summarization\artifacts\model\ts_model.pickle')

    
    def copy_weights(self,encoder,decoder):
        loaded_layer_names = [layer.name for _,layer in enumerate(self.trained_model.layers)]

        # ENCODER
        for layer in encoder.layers:
            if layer.name in loaded_layer_names:
                index = loaded_layer_names.index(layer.name)
                layer_weights = self.trained_model.layers[index].weights
            layer.set_weights(layer_weights)
        
        # DECODER
        for layer in decoder.layers:
            if layer.name in loaded_layer_names:
                index = loaded_layer_names.index(layer.name)
                layer_weights = self.trained_model.layers[index].get_weights()
                if layer_weights:
                    layer.set_weights(layer_weights)
        
        return encoder,decoder
                    


    def get_encoder_decoder(self):
    # CREATE ENCODER
        encoder = create_encoder(self.x_voc_size)
        encoder_outputs,state_h,state_c = encoder.call(self.encoder_inputs)
        context_vector = [state_h,state_c]
        encoder_model = Model(inputs=self.encoder_inputs,outputs=[encoder_outputs, state_h, state_c])
        
    # CREATE DECODER
        decoder = create_decoder(self.y_voc_size)
        decoder_output,decoder_h_state, decoder_c_state = decoder.call(self.decoder_inputs,context_vector)
        decoder_model = Model(inputs = [self.decoder_inputs] + context_vector, outputs=[decoder_output, decoder_h_state, decoder_c_state])
    
    # COPY WEIGHTS    
        encoder_model,decoder_model = self.copy_weights(encoder_model,decoder_model)

        return encoder_model,decoder_model
    

class get_tokenizer():
    def __init__(self):
        self.x_tokeizer_path = os.path.join(r"U:\nlp_project\text_summarization\artifacts\tokenizer\x" , 'x_tokenizer.pkl')
        self.y_tokenizer_path = os.path.join(r"U:\nlp_project\text_summarization\artifacts\tokenizer\y" , 'y_tokenizer.pkl')


    def get_tokenizers(self):
        with open(self.x_tokeizer_path, 'rb') as file:
            x_tokenizer = pickle.load(file)

        with open(self.y_tokenizer_path, 'rb') as file:
            y_tokenizer = pickle.load(file)
        
        return x_tokenizer,y_tokenizer