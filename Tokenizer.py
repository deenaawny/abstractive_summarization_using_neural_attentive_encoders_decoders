#Copied Code
import numpy as np
from keras_preprocessing.sequence import pad_sequences

class TokenText:

 def countRareWords(self, x_tokenizer):
   thresh=4

   cnt=0
   tot_cnt=0
   freq=0
   tot_freq=0

   for key,value in x_tokenizer.word_counts.items():
     tot_cnt=tot_cnt+1
     tot_freq=tot_freq+value
     if(value<thresh):
       cnt=cnt+1
       freq=freq+value

   return(tot_cnt,cnt)

 def convertTextSequencesInToIntegerSequences(self, x_tokenizer, x_tr, x_val, tot_cnt, cnt, max_text_len):
 #prepare a tokenizer for reviews on training data
#convert text sequences into integer sequences
   x_tr_seq    =   x_tokenizer.texts_to_sequences(x_tr)
   x_tr_seq_int = []
   for item in x_tr_seq:
     item_int = []
     for number in item:
      item_int.append(int(number))
     x_tr_seq_int.append(item_int)

   x_val_seq   =   x_tokenizer.texts_to_sequences(x_val)
   x_val_seq_int = []
   for item in x_val_seq:
    item_int = []
    for number in item:
     item_int.append(int(number))
    x_val_seq_int.append(item_int)

   # newest numpy Type error for floats
   # TypeError: 'numpy.float64' object cannot be interpreted as an integer
#padding zero upto maximum length
   x_tr    =  pad_sequences(x_tr_seq_int,  maxlen=int(max_text_len), padding='post')
   x_val   =  pad_sequences(x_val_seq_int, maxlen=int(max_text_len), padding='post')

#size of vocabulary ( +1 for padding token)
   x_voc   =  x_tokenizer.num_words + 1

   return(x_tr,x_val,x_voc)

#Here, I am deleting the rows that contain only START and END tokens
 def deleteRowsContainingStartAndEndTokens(self, x_tr, y_tr, x_val, y_val):
      ind=[]
      for i in range(len(y_tr)):
        cnt=0
        for j in y_tr[i]:
         if j!=0:
          cnt=cnt+1
         if(cnt==2):
          ind.append(i)

      y_tr=np.delete(y_tr,ind, axis=0)
      x_tr=np.delete(x_tr,ind, axis=0)

      ind=[]
      for i in range(len(y_val)):
        cnt=0
        for j in self.y_tr[i]:
          if j!=0:
            cnt=cnt+1
          if(cnt==2):
            ind.append(i)

      y_val=np.delete(y_val,ind, axis=0)
      x_val=np.delete(x_val,ind, axis=0)

      return(x_tr,x_val,y_tr,y_val)

# References
# [1] How to build own text summarizer using deep learning, computer code, downloaded 31 March 2020,
# < https://github.com/aravindpai/How-to-build-own-text-summarizer-using-deep-learning/blob/
# master/How_to_build_own_text_summarizer_using_deep_learning.ipynb>.

# -----------------------
# Important Information
# -----------------------
# fit_on_texts
# ------------
# fit_on_texts Updates internal vocabulary based on a list of texts.
# method creates the vocabulary index based on word frequency.
# So if you give it something like, "The cat sat on the mat."
# It will create a dictionary s.t. word_index["the"] = 1;
# word_index["cat"] = 2 it is word -> index dictionary
# so every word gets a unique integer value. 0 is reserved for padding.
# So lower integer means more frequent word (often the first few are stop words
# because they appear a lot).

# Keras Text Tokenizer
# -----------------------
#Text tokenization utility class.
#This class allows to vectorize a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary) or into a vector where the coefficient for each token could be binary, based on word count, based on tf-idf...
#Arguments
#num_words: the maximum number of words to keep, based on word frequency. Only the most common num_words-1 words will be kept.