#Copied Code
import re
from keras.callbacks import EarlyStopping
from keras_preprocessing.text import Tokenizer
from nltk.corpus import stopwords

from DataSet import DataSet
from DistributionOfSequences import DistributionOfSequences
from Document import Document

from glob import glob

from Encoder import Encoder
from Inference import Inference
from RougeMetric import RougeMetric
from Tokenizer import TokenText

from keras import backend as K

contraction_mapping = {"ain't": "is not", "aren't": "are not",
                       "can't": "cannot", "'cause": "because",
                       "could've": "could have", "couldn't": "could not",
                       "didn't": "did not", "doesn't": "does not",
                       "don't": "do not", "hadn't": "had not",
                       "hasn't": "has not", "haven't": "have not",
                       "he'd": "he would", "he'll": "he will", "he's": "he is",
                       "how'd": "how did", "how'd'y": "how do you",
                       "how'll": "how will", "how's": "how is",
                       "I'd": "I would", "I'd've": "I would have",
                       "I'll": "I will", "I'll've": "I will have",
                       "I'm": "I am", "I've": "I have", "i'd": "i would",
                       "i'd've": "i would have", "i'll": "i will",
                       "i'll've": "i will have", "i'm": "i am",
                       "i've": "i have", "isn't": "is not", "it'd": "it would",
                       "it'd've": "it would have", "it'll": "it will",
                       "it'll've": "it will have", "it's": "it is",
                       "let's": "let us", "ma'am": "madam",
                       "mayn't": "may not", "might've": "might have",
                       "mightn't": "might not", "mightn't've": "might not have",
                       "must've": "must have",
                       "mustn't": "must not", "mustn't've": "must not have",
                       "needn't": "need not", "needn't've": "need not have",
                       "o'clock": "of the clock",
                       "oughtn't": "ought not", "oughtn't've": "ought not have",
                       "shan't": "shall not", "sha'n't": "shall not",
                       "shan't've": "shall not have",
                       "she'd": "she would", "she'd've": "she would have",
                       "she'll": "she will", "she'll've": "she will have",
                       "she's": "she is",
                       "should've": "should have", "shouldn't": "should not",
                       "shouldn't've": "should not have", "so've": "so have",
                       "so's": "so as",
                       "this's": "this is", "that'd": "that would",
                       "that'd've": "that would have", "that's": "that is",
                       "there'd": "there would",
                       "there'd've": "there would have", "there's": "there is",
                       "here's": "here is", "they'd": "they would",
                       "they'd've": "they would have",
                       "they'll": "they will", "they'll've": "they will have",
                       "they're": "they are", "they've": "they have",
                       "to've": "to have",
                       "wasn't": "was not", "we'd": "we would",
                       "we'd've": "we would have", "we'll": "we will",
                       "we'll've": "we will have", "we're": "we are",
                       "we've": "we have", "weren't": "were not",
                       "what'll": "what will", "what'll've": "what will have",
                       "what're": "what are",
                       "what's": "what is", "what've": "what have",
                       "when's": "when is", "when've": "when have",
                       "where'd": "where did", "where's": "where is",
                       "where've": "where have", "who'll": "who will",
                       "who'll've": "who will have", "who's": "who is",
                       "who've": "who have",
                       "why's": "why is", "why've": "why have",
                       "will've": "will have", "won't": "will not",
                       "won't've": "will not have",
                       "would've": "would have", "wouldn't": "would not",
                       "wouldn't've": "would not have", "y'all": "you all",
                       "y'all'd": "you all would",
                       "y'all'd've": "you all would have",
                       "y'all're": "you all are", "y'all've": "you all have",
                       "you'd": "you would", "you'd've": "you would have",
                       "you'll": "you will", "you'll've": "you will have",
                       "you're": "you are", "you've": "you have"}

stop_words = set(stopwords.words('english'))

# Note: this text_cleaner removes numeric data
def text_cleaner(text, num):
  tcString = text.lower()
  # newString = BeautifulSoup(newString, "lxml").text
  tcString = re.sub(r'\([^)]*\)', '',  tcString)
  tcString = re.sub('"', '',  tcString)
  tcString = ' '.join(
      [contraction_mapping[t] if t in contraction_mapping else t for t in
       tcString.split(" ")])
  tcString = re.sub(r"'s\b", "",  tcString)
  tcString = re.sub("[^a-zA-Z]", " ",  tcString)
  tcString = re.sub('[m]{2,}', 'mm',  tcString)
  if (num == 0):
    tokens = [w for w in  tcString.split() if not w in stop_words]
  else:
    tokens =  tcString.split()
  long_words = []
  for i in tokens:
    if len(i) > 1:  # removing short word
      long_words.append(i)
  return (" ".join(long_words)).strip()

# text_cleaner example 1
cleanedText = text_cleaner("""A city trader who conned millions of pounds from wealthy investors was yesterday ordered to pay back £1.

Nicholas Levene, 48, was jailed for 13 years last November after he admitted orchestrating a lucrative Ponzi scheme which raked in £316million.

He used the money to finance his own lavish lifestyle with private jets, super yachts and round-the-world trips.
""", 1)

print(cleanedText)

def importAndCleanData(directory):
  filenames = glob(directory)
  datatextlists = []
  datasummarylists = []
  i = 1

  for f in filenames:
    if i == 10000:
      break
    d = Document(f)
    sentencesAppended = d.getSentencesAppended()
    summaryAppended = d.getSummaryAppended()
    datatextlists.append(sentencesAppended)
    datasummarylists.append(summaryAppended)
    i = i + 1

  cleaned_text = []
  for t in datatextlists:
   cleaned_text.append(text_cleaner(t, 0))

  cleaned_summary = []
  for t in datasummarylists:
    cleaned_summary.append(text_cleaner(t,1))

  return(cleaned_text,cleaned_summary)

def selectMaxLengthArticlesAndSummaries(data):
  ds = DistributionOfSequences()
  (max_text_len , max_summary_len ) =ds.maxLengthOfTheSequence(data)
  selectedArticlesAndSummaries = ds.selectArticlesAndSummaries(data,max_text_len,max_summary_len)
  selectedArticlesAndSummaries = ds.addStartAndEndTokens(selectedArticlesAndSummaries)

  return (selectedArticlesAndSummaries, max_text_len, max_summary_len)

def selectMaxArticlesAndSummariesAndConvertToSequences(data):
  (dataUnderMax, max_text_len, max_summary_len) = selectMaxLengthArticlesAndSummaries(data)
  ds = DataSet()
  (x_tr,x_val,y_tr,y_val)= ds.splitDataSet(dataUnderMax)
  t = TokenText()
  x_tokenizer = Tokenizer()
  x_tokenizer.fit_on_texts(list(x_tr))
  (xtot_cnt,xcnt)= t.countRareWords(x_tokenizer)
  y_tokenizer = Tokenizer()
  y_tokenizer.fit_on_texts(list(x_tr))
  (ytot_cnt,ycnt)= t.countRareWords(y_tokenizer)

  x_tokenizer = Tokenizer(num_words=xtot_cnt-xcnt)
  x_tokenizer.fit_on_texts(list(x_tr))
  (x_tr,x_val,x_voc) = t.convertTextSequencesInToIntegerSequences(x_tokenizer, x_tr, x_val, xtot_cnt, xcnt, max_text_len)
  y_tokenizer = Tokenizer(num_words=ytot_cnt-ycnt)
  y_tokenizer.fit_on_texts(list(y_tr))
  (y_tr,y_val,y_voc) = t.convertTextSequencesInToIntegerSequences(y_tokenizer, y_tr, y_val, ytot_cnt, ycnt, max_summary_len)

  return (max_text_len, max_summary_len, x_voc, y_voc, x_tr, y_tr, x_val, y_val, x_tokenizer, y_tokenizer)

def buildEncoderModelAndFit(max_text_len, x_voc, y_voc, x_tr, y_tr, x_val, y_val):
  encoder = Encoder(int(max_text_len), int(x_voc), int(y_voc))
  bm = encoder.buildModel(K)
  model = bm[0]
  encoder_inputs = bm[1]
  encoder_outputs = bm[2]
  state_h = bm[3]
  state_c = bm[4]
  latent_dim = bm[5]
  dec_emb_layer = bm[6]
  decoder_inputs = bm[7]
  decoder_lstm = bm[8]
  decoder_dense = bm[9]
  attn_layer  = bm[10]
  print(model.summary())
  model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)

  history=model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:] ,
                    epochs=50,callbacks=[es],batch_size=128, validation_data=([x_val,y_val[:,:-1]],
                                                                              y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))
  return(encoder_inputs, encoder_outputs,
         state_h, state_c, latent_dim, dec_emb_layer,
         decoder_inputs, decoder_lstm, decoder_dense, attn_layer,
         max_text_len)

def wordIndexes(x_tokenizer, y_tokenizer):
  reverse_source_word_index=x_tokenizer.index_word
  reverse_target_word_index=y_tokenizer.index_word
  target_word_index=y_tokenizer.word_index
  return(reverse_source_word_index, reverse_target_word_index, target_word_index)

def encoderDecoderSummarizationTest(directory):
  (cleaned_text,cleaned_summary) = importAndCleanData(directory)
  data= {'cleaned_text': cleaned_text, 'cleaned_summary': cleaned_summary}

  (max_text_len, max_summary_len, x_voc, y_voc, x_tr, y_tr, x_val, y_val, x_tokenizer, y_tokenizer) = selectMaxArticlesAndSummariesAndConvertToSequences(data)

  (encoder_inputs, encoder_outputs,
   state_h, state_c, latent_dim, dec_emb_layer,
   decoder_inputs, decoder_lstm, decoder_dense, attn_layer,
   max_text_len) = buildEncoderModelAndFit(max_text_len, x_voc, y_voc, x_tr, y_tr, x_val, y_val)

  (reverse_source_word_index, reverse_target_word_index, target_word_index) = wordIndexes(x_tokenizer,y_tokenizer)

  inference =Inference(encoder_inputs, encoder_outputs,
  state_h, state_c, latent_dim, dec_emb_layer,
  decoder_inputs, decoder_lstm, decoder_dense, attn_layer,
  target_word_index, reverse_source_word_index, reverse_target_word_index,
  int(max_text_len), int(max_summary_len))
  inference.encodeDecode()

  for i in range(0,len(x_tr)):
    print("Review:",inference.seq2text(x_tr[i]))
    os = inference.seq2summary(y_tr[i])
    print("Original summary:", os)
    ps = inference.decode_sequence(x_tr[i].reshape(1,int(max_text_len)))
    print("Predicted summary:", ps)
    #rm = RougeMetric()
    #rm.computeRouge([os], [ps])
    print("\n")


print("DailyMail Examples")
encoderDecoderSummarizationTest('C:/Users/admin/Documents/7lytixTest/dailymail_stories/dailymail/stories/*.story')

# References
# [1] How to build own text summarizer using deep learning, computer code, downloaded 31 March 2020,
# < https://github.com/aravindpai/How-to-build-own-text-summarizer-using-deep-learning/blob/
# master/How_to_build_own_text_summarizer_using_deep_learning.ipynb>.


