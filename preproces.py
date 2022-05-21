import nltk
import pandas as pd
import re
import string
from textblob import TextBlob
from nltk.corpus import stopwords
import emoji
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


# Intializing 
nltk.download('stopwords')
nltk.download('punkt')
remove = string.punctuation

def Preprocessing(df):
        remove_words = set(stopwords.words("english"))
        df["review"] = df["review"].apply(remove_html)
        df["review"] = df["review"].apply(remove_url)
        df["review"] = df["review"].apply(remove_punctuation)
        df["review"] = df["review"].apply(remove_stopwords,stopwords = remove_words)
        df["review"] = df["review"].apply(remove_emoji)
        df["review"] = df["review"].apply(blobTokenize)
        vector = TfidfVectorizer(max_features=500)
        review_vector = vector.fit_transform(df["review"]).toarray()
        return review_vector
    
    
	
def remove_html(text):
  pattern_html = re.compile('<.*?>')
  return pattern_html.sub(r'',text)

def remove_url(text):
  pattern_url = re.compile('https?://\S+|www\.\S+')
  return pattern_url.sub(r'',text)

def remove_punctuation(text):
  return text.translate(str.maketrans('','',remove))

def remove_stopwords(text, stopwords):
  return ' '.join([word for word in text.split() if word not in stopwords])

def remove_emoji(text):
  return emoji.demojize(text)

def blobTokenize(text):
    words = TextBlob(text).words
    return ','.join(words)


