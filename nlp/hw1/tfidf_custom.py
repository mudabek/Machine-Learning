import numpy as np
from collections import Counter
import math


class TfIdf():
    def __init__(self, train_text):
        self.idf_dict = fit_vectorizer(train_text)

    def transform(self, data):
        transformed = []
        for row in data:
            transformed.append(vectorize(row, self.idf_dict))
        return np.array(transformed)


def compute_tf_dict(text):
  """function to compute TF for a given text"""
  tf_dict = {}
  # count word occurence in a text
  for word in text.split():
    if word in tf_dict:
      tf_dict[word] += 1
    else:
      tf_dict[word] = 1
  # compute tf for each word
  for word in tf_dict:
    tf_dict[word] = tf_dict[word] / len(text.split())
  return tf_dict


def compute_count_dict(data):
  """function to compute all word occurences in a dataset"""
  count_dict = {}
  for text in data:
    set_of_words = set(text.split())
    for word in set_of_words:
      if word in count_dict:
        count_dict[word] += 1
      else:
        count_dict[word] = 1
  return count_dict


def compute_idf_dict(data):
  """compute idf values for words in dataset"""
  count_dict = compute_count_dict(data)
  N = len(data)
  idf_dict = {}
  for word in count_dict:
    idf_dict[word] = math.log(N / count_dict[word])
  return idf_dict


def fit_vectorizer(data):
  """get the idf values for a given dataset"""
  idf_dict = compute_idf_dict(data)
  return idf_dict


def vectorize(text, idf_dict):
  """cast a given text to a tf_idf values"""
  token_tdidfs = np.zeros(len(idf_dict))
  tf_dict = compute_tf_dict(text)
  text_list = text.split()
  for i, word in enumerate(idf_dict):
    if word in text_list:
      token_tdidfs[i] = tf_dict[word] * idf_dict[word]
  return np.array(token_tdidfs, 'float32')
