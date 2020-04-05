#Copied Code
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class DistributionOfSequences:

 def maxLengthOfTheSequence(self,data):
   text_word_count = []
   summary_word_count = []

   for i in data['cleaned_text']:
    text_word_count.append(len(i.split()))

   for i in data['cleaned_summary']:
    summary_word_count.append(len(i.split()))

   print('\nNumpy Array\n----------\n', text_word_count)
   x,y,z = plt.hist(bins=30, x= text_word_count)
   plt.show()
   bin_max = np.where(x == x.max())

   print('\nNumpy Array\n----------\n', text_word_count)
   x2,y2,z2 = plt.hist(bins=30, x= summary_word_count)
   plt.show()
   bin_max2 = np.where(x == x.max())
   max_text_len = y[bin_max][0]
   max_summary_len = y2[bin_max2][0]

   return (max_text_len , max_summary_len )

 def selectArticlesAndSummaries(self,data, max_text_len, max_summary_len):
   cleaned_text =np.array(data['cleaned_text'])
   cleaned_summary=np.array(data['cleaned_summary'])

   short_text=[]
   short_summary=[]

   for i in range(len(cleaned_text)):
     if(len(cleaned_summary[i].split())<= max_summary_len and len(cleaned_text[i].split())<= max_text_len):
       short_text.append(cleaned_text[i])
       short_summary.append(cleaned_summary[i])

   selecteddata=pd.DataFrame({'text':short_text,'summary':short_summary})
   return(selecteddata)

 def addStartAndEndTokens(self, data):
    data['summary'] =  ['sostok '+ x + ' eostok' for x in data['summary']]
    return data

# References
# [1] How to build own text summarizer using deep learning, computer code, downloaded 31 March 2020,
# < https://github.com/aravindpai/How-to-build-own-text-summarizer-using-deep-learning/blob/
# master/How_to_build_own_text_summarizer_using_deep_learning.ipynb>.