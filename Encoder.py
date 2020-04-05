#Copied Code
from keras import Input, Model
from keras.layers import Embedding, LSTM, Concatenate, TimeDistributed, Dense

from Attention import AttentionLayer

class Encoder:

  def __init__(self, max_text_len, x_voc, y_voc):
    self.max_text_len = max_text_len
    self.x_voc = x_voc
    self.y_voc = y_voc

  def buildModel(self, K):
    K.clear_session()
    latent_dim = 300
    embedding_dim = 100

    # Encoder
    encoder_inputs = Input(shape=(self.max_text_len,))
    #embedding layer
    enc_emb =  Embedding(self.x_voc, embedding_dim,trainable=True)(encoder_inputs)

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
    dec_emb_layer = Embedding(self.y_voc, embedding_dim,trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
    decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

    # Attention layer
    attn_layer = AttentionLayer(K,name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

    # Concat attention input and decoder LSTM output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

    #dense layer
    decoder_dense =  TimeDistributed(Dense(self.y_voc, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_concat_input)

    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return [ model, encoder_inputs, encoder_outputs,
             state_h, state_c, latent_dim,
             dec_emb_layer, decoder_inputs, decoder_lstm,
             decoder_dense, attn_layer]
# References
# [1] How to build own text summarizer using deep learning, computer code, downloaded 31 March 2020,
# < https://github.com/aravindpai/How-to-build-own-text-summarizer-using-deep-learning/blob/
# master/How_to_build_own_text_summarizer_using_deep_learning.ipynb>.