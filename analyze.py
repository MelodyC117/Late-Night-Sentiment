#########################################################################################################
""" This script was solely used for DATA ANALYSIS to analyze cleaned tweet data
    Analysis include -
        x. Frequent itemset mining with FP Growth
        x. Topic Mining with LDA
        x. Sentiment Analysis with TextBlob
        x. Hypothesis Testing with ttest """

import pyfpgrowth
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pandas as pd
import numpy as np
import gensim
from gensim import corpora
import nltk
from scipy import stats
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from textblob import TextBlob
import spacy
nlp = spacy.load('en_core_web_sm')
nlp1 = spacy.load('en_core_web_lg')
import frequency

################################################## FREQUENCY #############################################

" count word frequency"
def word_frequency(df):

    # use the flatten function from clean.py and generate word frequency
    transaction = frequency(df['interpret'])
    counts = collections.Counter(transaction)

    return counts.most_common(50)


"mining frequent itemset"
def growth(transaction, min_sup):

    # use min_sup as a percentage
    patterns = pyfpgrowth.find_frequent_patterns(transaction, min_sup*len(transaction))
    # sort items with descending item counts
    organized = sorted([(i, k) for (i, k) in patterns.items()], key = lambda x: -x[1])

    return organized


################################################ LDA CLUSTERING ###########################################

"convert words to bigrams"
def prepare(transaction, min_count, threshold):

    # convert words to bigrams to enable statistical analysis of co-occurrence
    # all functions were enabled and given by the LDA library
    b_= gensim.models.Phrases(transaction, min_count=min_count, threshold=threshold)
    bigram = gensim.models.phrases.Phraser(bigram)

    return bigram


"lemmatize words"
def lemmatize(transaction, min_count, threshold, postag):
    bigram = prepare(transaction, min_count, threshold)
    process = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    bigram_list, output = [], []
    for word in transactions:
        bigram_list.append(bigram[word])

    for br in bigram_list:
        tokens = process(" ".join(br))
        for token in tokens:
            if token.pos_ in postag:
                output.append(token.lemma_)

    return output


"get corpus from lemmatized word list"
def get_corpus(lemmatized):

    id2word = corpora.Dictionary(lemmatized)
    corpus = [id2word.doc2bow(word) for word in lemmatized]

    return id2word, corpus


"run LDA model for lemmetized word list"
def LDA(corpus, id2word, num_topics):

    # the lda object contains many metrics that are not directly displayed
    # use additional functions below to extract metrics as needed
    lda_object = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics,
                                                 random_state=100, update_every=1, chunksize=100,
                                                 passes=10, alpha='auto', per_word_topics=True)

    return lda_object

"extract output from lda object"
def print_corpus(lda_object):

    # outputs tuples of (cluster number, probability, words) combination
    # e.g. (0, '0.050*"year" + 0.048*"day" + 0.016*"hour" + 0.013*"family" + 0.012*"job")
    print(lda_object.print_topics())


"associate original word list to their clustered topics"
def check_topics(lda_object, corpus, transaction):

    # topic displays LDA clustered topics
    # tweet displays tokenized word lists from tweets
    organize = pd.DataFrame(columns=['topic', 'tweet'])

    for num, rows in enumerate(lda_object[corpus]):
        row = rows[0] if lda_object.per_word_topics else rows
        row = sorted(row, key=lambda x: x[1], reverse=True)

        for i, topic in enumerate(row):
            temp = []
            if i == 0:
                topic_list = lda_object.show_topic(topic[0])

                for word, p in topic_list:
                    temp.append(word)
                organize['topic'] = temp
            else:
                break

    organize['tweet'] = transaction

    return organize


"extract only clusters and topics"
def extract_topic(lda_object):
    topics, w = [], []
    for topic in lda_object:
        cluster, words = topic
        for word in words:
            w.append(word)
        topics.append((topic, w))

    for i, j in topics:
        print("cluster:", i, "topics:", " ".join(j))

############################################## SENTIMENT ANALYSIS #########################################


"calculate polarity and subjectivity for tweets"
def add_sentiment(df):

    # apply sentiment score to raw tweets
    polarity = lambda x: TextBlob(x).sentiment.polarity
    subjectivity = lambda x: TextBlob(x).sentiment.subjectivity
    df['polarity'] = df.tweet.apply(polarity)
    df['subjectivity'] = df.tweet.apply(subjectivity)

    def classify(x):
        if x > 0:
            return 'Positive'
        elif x == 0:
            return 'Neutral'
        else:
            return 'Negative'

    def score(x):
        if x > 0:
            return 1
        elif x == 0:
            return 0
        else:
            return -1

    # add labels and score for polarity ranges using subfunctions above
    df['classify'] = df.polarity.apply(classify)
    df['score'] = df.polarity.apply(score)

    return df


"compare sentiment scores between day and late night"
def sent_by_timeframe(df, criteria, sentiment):

    # given criteria = measured by [mean, min, max]
    # given sentiment categories = [polarity, subjectivity]
    # return grouped object to be processed later
    if sentiment == "p":
        if criteria == "mean":
            group = df.groupby('late').polarity.mean()
        elif criteria == "min":
            group = df.groupby('late').polarity.min()
        else:
            group = df.groupby('late').polarity.max()
    else:
        if criteria == "mean":
            group = df.groupby('late').subjectivity.mean()
        elif criteria == "min":
            group = df.groupby('late').subjectivity.min()
        else:
            group = df.groupby('late').subjectivity.max()

    return group


"compare mean sentiment scores for each hour of the night"
def sent_by_hour(df, sentiment):

    # given sentiment categories = [polarity, subjectivity]
    # return grouped object to be processed later
    if sentiment == "p":
        group = df.groupby('hour').polarity.mean()
    else:
        group = df.groupby('hour').subjectivity.mean()

    return group

############################################## HYPOHESIS TESTING ##########################################


"prepare observation and experiment series for hypothesis testing"
def process_for_test(df):

    # unstack and reset the group index to make the format readable
    # e.g. late	   user_id	     N	        Y	      diff
          # 0	  10085872	 0.010000	0.250000	0.240000
          # 2	  10283432	 0.063056	0.353571	0.290516
    df = df.unstack().reset_index()
    df.dropna(inplace=True)
    df['diff'] = df['Y'] - df['N']

    observation = df.Y.values.tolist()
    experiment = df.N.values.tolist()

    return observation, experiment


"perform pairwise ttest with 95\% \confidence level"
def ttest(observation, experiment):

    # output e.g. Ttest_relResult(statistic=0.975347447335138, pvalue=0.0397876645022356)
    print(stats.ttest_rel(observation, experiment))

#########################################################################################################
