
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score



train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv') # load data from csv
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv') # load data from csv
submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

vocab=train.text.append(test.text,ignore_index=True)
vocab_copy=train.text.append(test.text,ignore_index=True)
vocab[:10]

from nltk.corpus import stopwords
import nltk
from gensim.corpora import Dictionary
from keras.preprocessing.sequence import pad_sequences
from nltk.stem import PorterStemmer,LancasterStemmer
from nltk.util import ngrams

stemmer=PorterStemmer()
lan_stemmer=LancasterStemmer('english')

from sklearn.decomposition import PCA



vocab_list=vocab.tolist()

remove_text='https?'
vocab=vocab.str.replace(remove_text,'')
vocab=vocab.str.findall('[a-zA-Z]+')
stopword=stopwords.words('english')

vocab=vocab.apply(lambda x: [w.lower() for w in x if w.lower() not in stopword])
vocab=vocab.apply(lambda x: [stemmer.stem(w) for w in x])
vocab[:10]

dictionary=Dictionary(vocab)

train.text=vocab.iloc[:7613]

test.text=vocab[7613:].reset_index().text

train.text=train.text.apply(lambda x: dictionary.doc2idx(x))
train_text=pad_sequences(train.text)
train_target=train.target.values

from gensim.models.word2vec import Word2Vec

import inspect
word2vec=Word2Vec(vocab,size=32)
#inspect.signature(Word2Vec)

vocab[:10]

word_vectors=word2vec.wv.vectors

word_vectors.shape

pca=PCA(n_components=2)
inspect.signature(PCA)

pca.fit(word_vectors)

pca.transform(word_vectors).shape

word2vec.

train.text.apply(lambda x: (map(x,dictionary.token2id)))

list(map(dictionary.token2id.get,['allah']))

dictionary.token2id

test.text=test.text.apply(lambda x: dictionary.doc2idx(x))

test_text=pad_sequences(test.text)
#test_target=test.target.values

test_text.shape,train_text.shape

len(dictionary.keys())

from sklearn.model_selection import train_test_split
train_x,val_x,train_y,val_y=train_test_split(train_text,train_target,test_size=.2)

from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout,LSTM

vocab_size=len(dictionary.keys())
input_dimension=train_x.shape[1]



def create_model():
    model=Sequential()
    model.add(Embedding(vocab_size,16,input_length=input_dimension))

    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


model = KerasClassifier(build_fn=create_model)
grid_model=GridSearchCV(model,cv=10,param_grid={'epochs':[2]},scoring='accuracy')

grid_model.fit(train_x,train_y)

grid_model.score(val_x,val_y),grid_model.best_params_

import math

math.pow(vocab_size,1/4)

y_test_predicted=grid_model.predict(test_text)

y_test_predicted.flatten()

test['predicted']=y_test_predicted

submission.target=test.predicted

from datetime import datetime

datetime.today()

submission.to_csv('predicted_{}.csv'.format(datetime.today()),index=False)

train.text.apply(lamddbda x: [nltk.pos_tag(x)])

train['tweet_len'] = train['text'].astype(str).apply(len)
train['word_count'] = train['text'].apply(lambda x: len(str(x).split()))

# Commented out IPython magic to ensure Python compatibility.
from plotly import __version__
# %matplotlib inline
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
init_notebook_mode(connected=True)

init_notebook_mode(connected=True)
cf.go_offline()
train['tweet_len'].iplot(
    kind='hist',
    bins=100,
    xTitle='tweet length',
    linecolor='black',
    yTitle='count',
    title='Tweet Length Distribution')

import matplotlib.pyplot as plt
neg_tweets = train[train.target == 0]
neg_string = []
for t in neg_tweets.text:
    neg_string.append(t)
neg_string = pd.Series(neg_string).str.cat(sep=' ')
from wordcloud import WordCloud

wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

import matplotlib.pyplot as plt
pos_tweets = train[train.target == 1]
pos_string = []
for t in pos_tweets.text:
    pos_string.append(t)
pos_string = pd.Series(pos_string).str.cat(sep=' ')
from wordcloud import WordCloud

wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

train['word_count'].iplot(
    kind='hist',
    bins=100,
    xTitle='word count',
    linecolor='black',
    yTitle='count',
    title='Tweet Word Count Distribution')

train['hastags'] = train['text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
train[['text','hastags']].head()

#function to extract @mentions and #tags
def extracter():
    mentions={}
    tags={}
    for i in df_trend_user.index:
        tokens = df_trend_user['text'][i].split()    
        for token in tokens:
            if('@' in token[0] and len(token) > 1):
                if token.strip('@') in mentions:
                    mentions[token.strip('@')] += 1
                else:
                    mentions[token.strip('@')] = 1
        
        
            if('#' in token[0] and len(token) > 1):
                if token.strip('#') in tags:
                    tags[token.strip('#')] += 1
                else:
                    tags[token.strip('#')] = 1    
                    
    return mentions,tags

df_trend_user = train.loc[:,['text']]

mentions ,tags = extracter()

mentions_keys = list(mentions.keys())
mentions_values = list(mentions.values())
tags_keys = list(tags.keys())
tags_values = list(tags.values())

#barplot function
import seaborn as sns
def drawbarplot(x,y,xlabel,title,figsize=(10,10)):
    plt.figure(figsize=figsize)    
    sns.barplot(x=x,y=y,palette = 'terrain',orient='h',order=y)
    for i,v in enumerate(x):
        plt.text(0.8,i,v,color='k',fontsize=10)
    
    plt.title(title,fontsize=20)
    plt.xlabel(xlabel,fontsize =14)
    plt.show()

df_mention = pd.DataFrame(columns=['mentions','m_count'])
df_mention['mentions'] = mentions_keys
df_mention['m_count'] = mentions_values
df_mention.sort_values(ascending=False,by='m_count',inplace=True)
df_count = df_mention.iloc[:50,:]
drawbarplot(x=df_count.m_count,y=df_count.mentions,xlabel='Count of mentions',title='Top 50 mentions',figsize=(16,16))

df_tags =pd.DataFrame(columns=['tags','t_count'])
df_tags['tags'] = tags_keys
df_tags['t_count'] = tags_values
df_tags.sort_values(ascending=False,by='t_count',inplace=True)
df_count = df_tags.iloc[:50,:]
drawbarplot(x=df_count.t_count,y=df_count.tags,xlabel='Count of tags',title='Top 50 Tags',figsize=(16,16))

train['hastags'].iplot(
    kind='hist',
    bins=100,
    xTitle='hashtags count',
    linecolor='black',
    yTitle='count',
    title='Tweet hashtags Count Distribution')

train['numerics'] = train['text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
train[['text','numerics']].head()

train['numerics'].iplot(
    kind='hist',
    bins=50,
    xTitle='numerics count',
    linecolor='black',
    yTitle='count',
    title='numeric Count Distribution')

train['upper'] = train['text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
train[['text','upper']].head()

train['upper'].iplot(
    kind='hist',
    bins=100,
    xTitle='numerics count',
    linecolor='black',
    yTitle='count',
    title='upper word Count Distribution')



train['upper'] = train['text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
train[['text','upper']].head()



from sklearn.feature_extraction.text import CountVectorizer
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(train['text'], 20)
for word, freq in common_words:
    print(word, freq)
df1 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
df1.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in tweet before removing stop words')

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(train['text'], 20)
for word, freq in common_words:
    print(word, freq)
df2 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
df2.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in tweet after removing stop words')

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(train['text'], 20)
for word, freq in common_words:
    print(word, freq)
df3 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
df3.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in tweet before removing stop words')

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(train['text'], 20)
for word, freq in common_words:
    print(word, freq)
df4 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
df4.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in review after removing stop words')

def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(train['text'], 20)
for word, freq in common_words:
    print(word, freq)
df5 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
df5.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams in tweet before removing stop words')

def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(train['text'], 20)
for word, freq in common_words:
    print(word, freq)
df6 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
df6.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams in tweet after removing stop words')

from textblob import TextBlob
blob = TextBlob(str(train['text']))
pos_df = pd.DataFrame(blob.tags, columns = ['word' , 'pos'])
pos_df = pos_df.pos.value_counts()[:20]
pos_df.iplot(
    kind='bar',
    xTitle='POS',
    yTitle='count', 
    title='Top 20 Part-of-speech tagging for tweet corpus')

#lowercase conversion
train['text'] = train['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
train['text'].head()

train['text'] = train['text'].str.replace('[^\w\s]','')
train['text'].head()

from nltk.corpus import stopwords
stop = stopwords.words('english')
train['text'] = train['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
train['text'].head()

freq = pd.Series(' '.join(train['text']).split()).value_counts()[:10]
freq

freq = pd.Series(' '.join(train['text']).split()).value_counts()[-30:]
freq

freq = list(freq.index)
train['text'] = train['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
train['text'].head()

from textblob import TextBlob
train['text']=train['text'].apply(lambda x: str(TextBlob(x).correct()))

from sklearn.base import BaseEstimator, TransformerMixin
class CleanText(BaseEstimator, TransformerMixin):
    def remove_mentions(self, input_text):
        return re.sub(r'@\w+', '', input_text)
    
    def remove_urls(self, input_text):
        return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)
    
    def emoji_oneword(self, input_text):
        # By compressing the underscore, the emoji is kept as one word
        return input_text.replace('_','')
    
    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
        return input_text.translate(trantab)

    def remove_digits(self, input_text):
        return re.sub('\d+', '', input_text)
    
    def to_lower(self, input_text):
        return input_text.lower()
    
    def remove_stopwords(self, input_text):
        #stopwords_list = st_words
        stopwords_list=STOPWORDS
        # Some words which might indicate a certain sentiment are kept via a whitelist        
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 1] 
        return " ".join(clean_words) 
    
    def stemming(self, input_text):
        porter = PorterStemmer()
        words = input_text.split() 
        stemmed_words = [porter.stem(word) for word in words]
        return " ".join(stemmed_words)
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.stemming)        
        return clean_X

import re
import emoji
class TextCounts(BaseEstimator, TransformerMixin):
    
    def count_regex(self, pattern, tweet):
        return len(re.findall(pattern, tweet))
    
    def fit(self, X, y=None, **fit_params):
        # fit method is used when specific operations need to be done on the train data, but not on the test data
        return self
    
    def transform(self, X, **transform_params):
        count_words = X.apply(lambda x: self.count_regex(r'\w+', x)) 
        count_mentions = X.apply(lambda x: self.count_regex(r'@\w+', x))
        count_hashtags = X.apply(lambda x: self.count_regex(r'#\w+', x))
        #count_capital_words = X.apply(lambda x: self.count_regex(r'\b[A-Z]{2,}\b', x))
        #count_excl_quest_marks = X.apply(lambda x: self.count_regex(r'!|\?', x))
        count_urls = X.apply(lambda x: self.count_regex(r'http.?://[^\s]+[\s]?', x))
        # We will replace the emoji symbols with a description, which makes using a regex for counting easier
        # Moreover, it will result in having more words in the tweet
        count_emojis = X.apply(lambda x: emoji.demojize(x)).apply(lambda x: self.count_regex(r':[a-z_&]+:', x))
        
        df = pd.DataFrame({'count_words': count_words
                           , 'count_mentions': count_mentions
                           , 'count_hashtags': count_hashtags                           
                           , 'count_urls': count_urls
                           , 'count_emojis': count_emojis
                          })
        
        return df

tc = TextCounts()
df_feature =  tc.fit_transform(train['text'])
df_feature.head(10)

ct = CleanText()
df_final['TweetBody'] = ct.fit_transform(train.text)
#Imputing '[no text]' value where there is no text
train.loc[train['text'] == '','TweetBody'] = '[no text]'