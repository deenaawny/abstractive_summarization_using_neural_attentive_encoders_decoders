#Copied Code
from keras import Model, Input
from keras.layers import Concatenate
from pandas import np

class Inference:

  def __init__(self, encoder_inputs, encoder_outputs,
      state_h, state_c, latent_dim, dec_emb_layer,
      decoder_inputs, decoder_lstm, decoder_dense, attn_layer,
      target_word_index, reverse_source_word_index, reverse_target_word_index,
      max_text_len, max_summary_len):
    self.encoder_inputs = encoder_inputs
    self.encoder_outputs = encoder_outputs
    self.state_h = state_h
    self.state_c = state_c
    self.latent_dim = latent_dim
    self.max_text_len = max_text_len
    self.dec_emb_layer = dec_emb_layer
    self.decoder_inputs = decoder_inputs
    self.decoder_lstm = decoder_lstm
    self.attn_layer = attn_layer
    self.decoder_dense = decoder_dense
    self.target_word_index = target_word_index
    self.reverse_source_word_index = reverse_source_word_index
    self.reverse_target_word_index = reverse_target_word_index
    self.max_summary_len = max_summary_len
    self.encoder_model = None
    self.decoder_model = None

  def encodeDecode(self):
    # Encode the input sequence to get the feature vector
    self.encoder_model = Model(inputs=self.encoder_inputs,outputs=[self.encoder_outputs, self.state_h, self.state_c])

    # Decoder setup
    # Below tensors will hold the states of the previous time step
    decoder_state_input_h = Input(shape=(self.latent_dim,))
    decoder_state_input_c = Input(shape=(self.latent_dim,))

    decoder_hidden_state_input = Input(shape=(self.max_text_len,self.latent_dim))

    # Get the embeddings of the decoder sequence
    dec_emb2= self.dec_emb_layer(self.decoder_inputs)
    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_outputs2, state_h2, state_c2 = self.decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

    #attention inference
    attn_out_inf, attn_states_inf = self.attn_layer([decoder_hidden_state_input, decoder_outputs2])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

    # A dense softmax layer to generate prob dist. over the target vocabulary
    decoder_outputs2 = self.decoder_dense(decoder_inf_concat)

    # Final decoder model
    self.decoder_model = Model(
    [self.decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])

  def decode_sequence(self, input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = self.encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = self.target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:

      output_tokens, h, c = self.decoder_model.predict([target_seq] + [e_out, e_h, e_c])

      # Sample a token
      # sampled_token_index = 0
     # while sampled_token_index == 0:
      sampled_token_index = np.argmax(output_tokens[0, -1, :])
      if sampled_token_index == 0:
        sampled_token_index = 1
      sampled_token = self.reverse_target_word_index[sampled_token_index]

      if(sampled_token!='eostok'):
        decoded_sentence += ' '+sampled_token

      # Exit condition: either hit max length or find stop word.
      if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (self.max_summary_len-1)):
        stop_condition = True

      # Update the target sequence (of length 1).
      target_seq = np.zeros((1,1))
      target_seq[0, 0] = sampled_token_index

      # Update internal states
      e_h, e_c = h, c

    return decoded_sentence

  def seq2summary(self, input_seq):
    newString=''
    for i in input_seq:
      if((i!=0 and i!=self.target_word_index['sostok']) and i!=self.target_word_index['eostok']):
        newString= newString + self.reverse_target_word_index[i] + ' '
    return newString

  def seq2text(self, input_seq):
    newString=''
    for i in input_seq:
      if(i!=0):
        newString= newString + self.reverse_source_word_index[i] + ' '
    return newString

# References
# [1] How to build own text summarizer using deep learning, computer code, downloaded 31 March 2020,
# < https://github.com/aravindpai/How-to-build-own-text-summarizer-using-deep-learning/blob/
# master/How_to_build_own_text_summarizer_using_deep_learning.ipynb>.