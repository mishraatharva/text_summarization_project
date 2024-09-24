import streamlit as st
from utils import Create_Encoder_Decoder,get_tokenizer
from prediction_pipeline import make_prediction
import pandas as pd
import tensorflow as tf

from gpusetup import gpu_setup
tf.debugging.set_log_device_placement(True)

gpu_setup()
device_spec = tf.DeviceSpec(job ="localhost", replica = 0, device_type = "GPU")

import streamlit as st


# Title of the Streamlit app
st.title("Text Summarization")

# Description of the app
st.write("This app uses NLP models to summarize large pieces of text. Enter the text you want to summarize below.")
text_input = st.text_area("Enter Text to Summarize", height=300)

@st.cache_resource
def load_summarizer():
    ec_obj = Create_Encoder_Decoder()
    encoder,decoder = ec_obj.get_encoder_decoder()
    tobj = get_tokenizer()
    x_tokenizer,y_tokenizer = tobj.get_tokenizers()
    return encoder,decoder,x_tokenizer,y_tokenizer


with tf.device(device_spec):
    encoder,decoder,x_tokenizer,y_tokenizer = load_summarizer()
    prediction = make_prediction(x_tokenizer,y_tokenizer,encoder,decoder)

    if st.button("Summarize Text"):
        if text_input:
            result = prediction.predict(text_input)
            # st.write(result)
            st.text_area("Result",value=result, height=300)
        else:
            st.write("Please enter some thing to predict!")
    
# Footer
st.write("Powered by LSTM and Streamlit")