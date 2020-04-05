#Copied Code
import numpy as np
from sklearn.model_selection import train_test_split

class DataSet:

  def splitDataSet(self,data):
    x_tr,x_val,y_tr,y_val=train_test_split(np.array(data['text']), np.array(data['summary']), test_size=0.1, random_state=0, shuffle=True)
    return(x_tr,x_val,y_tr,y_val)

# References
# [1] How to build own text summarizer using deep learning, computer code, downloaded 31 March 2020,
# < https://github.com/aravindpai/How-to-build-own-text-summarizer-using-deep-learning/blob/
# master/How_to_build_own_text_summarizer_using_deep_learning.ipynb>.