import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

news_data_preprocess = pd.read_csv('PreDataTestingDocSim1.csv')
text_cols = news_data_preprocess.select_dtypes(include=['object']).columns
news_data_preprocess[text_cols] = news_data_preprocess[text_cols].astype(str)


def document_similarity(input_string):
  # preprocessing function
  def preprocess_text(text):
      # case folding
      text = text.lower()

      # punctuation removal
      text = text.translate(str.maketrans('', '', string.punctuation))

      # tokenization
      tokens = word_tokenize(text)

      # # Bigram
      bigram = list(nltk.bigrams(tokens))
      bigram_list = [' '.join(bg) for bg in bigram]

      # stopword removal
      stop_words = set(stopwords.words('indonesian'))
      filtered_tokens = [token for token in tokens if not token in stop_words]

      # stemming
      stemmer = StemmerFactory().create_stemmer()
      stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

      # lemmatization
      lemmatizer = WordNetLemmatizer()
      lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]

      return ' '.join(lemmatized_tokens)

  # UNTUK PREPROSESS

  # news_data_preprocess['judul_preprocessed'] = news_data_preprocess['judul'].apply(preprocess_text)

  # create TF-IDF vectorizer object
  tfidf_vectorizer = TfidfVectorizer()

  # fit and transform the dataset
  tfidf_matrix = tfidf_vectorizer.fit_transform(news_data_preprocess['judul_preprocessed'])

  # take input from user
  input_text = input_string
  input_text = preprocess_text(input_text)

  # transform input text using TF-IDF vectorizer
  input_tfidf = tfidf_vectorizer.transform([input_text])

  # calculate cosine similarity between input text and dataset
  cosine_similarities = cosine_similarity(input_tfidf, tfidf_matrix).flatten()

  # get the indices of the most similar news articles
  most_similar_indices = cosine_similarities.argsort()[:-6:-1]

  # print('\nBerita tersebut memiliki kemiripan dengan berita..\n')
  num_true = 0
  num_fake = 0

  # DATASET PRERPROCESSED
  index = 0
  for index in most_similar_indices:
    if round(cosine_similarities[index],2) > 0.6:
      if news_data_preprocess.loc[index, 'label'] == 1:
          
          print('Judul berita:', news_data_preprocess.loc[index, 'judul'])
          print('Url berita:', news_data_preprocess.loc[index, 'url'])
          print('Label berita: True')
          print('Persentase kemiripan:', round(cosine_similarities[index]* 100, 2), '%\n')
          # print('Persentase kemiripan:', round(cosine_similarities[index],2))
          num_true += 1
          return '1',news_data_preprocess.loc[index, 'judul'], news_data_preprocess.loc[index, 'url'], 'True', round(cosine_similarities[index]* 100, 2), '%\n'
          
        
      elif news_data_preprocess.loc[index, 'label'] == 0:
          print('Judul berita:', news_data_preprocess.loc[index, 'judul'])
          print('Url berita:', news_data_preprocess.loc[index, 'url'])
          print('Label berita: Fake')
          print('Persentase kemiripan:', round(cosine_similarities[index]*100, 2), '%\n')
          # print('Persentase kemiripan:', round(cosine_similarities[index],2), '%\n')
          num_fake += 1
          return '1',news_data_preprocess.loc[index, 'judul'], news_data_preprocess.loc[index, 'url'], 'False', round(cosine_similarities[index]* 100, 2), '%\n'
      break
        
    else:
      return '0','Data berita tidak ditemukan atau coba tuliskan judul berita lebih lengkap'
      break
  
  