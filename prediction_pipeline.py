import numpy as np
from constants import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords


class make_prediction():
    def __init__(self,x_tokenizer,y_tokenizer,encoder,decoder):
        self.x_tokenizer = x_tokenizer
        self.y_tokenizer = y_tokenizer
        self.reverse_source_word_index = x_tokenizer.index_word
        self.reverse_target_word_index = y_tokenizer.index_word 
        self.target_word_index = y_tokenizer.word_index
        self.stop_words = set(stopwords.words('english'))
        self.encoder = encoder
        self.decoder = decoder


    def decode_sequence(self,input_seq):
        e_out, e_h, e_c = self.encoder.predict(input_seq)
        context_vector = [e_h, e_c]
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = self.target_word_index['start']

        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder.predict([target_seq] + [e_h, e_c]) # inputs = [decoder_inputs] + context_vector
            sampled_token_index = np.argmax(output_tokens[0, -1, :])

            if sampled_token_index == 0:
                stop_condition = True
            else:
                sampled_token = self.reverse_target_word_index[sampled_token_index]

            if(sampled_token!='end'):
                decoded_sentence += ' '+sampled_token
                # print(decoded_sentence)
                if (sampled_token == 'end' or len(decoded_sentence.split()) >= (MAX_LEN_SUMMARY-1)):
                    stop_condition = True
            target_seq = np.zeros((1,1))
            target_seq[0, 0] = sampled_token_index
            e_h, e_c = h, c

        return decoded_sentence

    
    def preprocess_text(self,text):
        text = text.lower()
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub('"','', text)
        text = ' '.join([CONTRACTION_MAPPING[t] if t in CONTRACTION_MAPPING else t for t in text.split(" ")])
        tokens = [w for w in text.split() if not w in self.stop_words]
        long_words = []
        for i in tokens:
            if len(i)>=3:
                long_words.append(i)
        text  = (" ".join(long_words)).strip()
        text = self.x_tokenizer.texts_to_sequences([text]) #  text = x_tokenizer.texts_to_sequences([text])
        text = pad_sequences(text, maxlen=MAX_LEN_TEXT, padding='post')
        return text
    
    def predict(self,text):
        text = self.x_tokenizer.texts_to_sequences([text])
        text = pad_sequences(text, maxlen=MAX_LEN_TEXT, padding='post')
        result = self.decode_sequence(text[0].reshape(1,MAX_LEN_TEXT))
        return result