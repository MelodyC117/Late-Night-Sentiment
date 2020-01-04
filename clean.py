#########################################################################################################
""" This script was solely used for DATA PREPROCESSING to prepare for text mining
    Cleaning steps include adding necessarily columns, break down date and time into additional categories,
    remove "dirty" features from raw texts, tokenize text strings to word lists, etc. """

import pandas as pd
import numpy as np
import os
import itertools
import collections
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
import re
import networkx
import warnings
warnings.filterwarnings("ignore")


"process time columns"
def add_field(df):
    # determine late night tweets
    df['late'] = np.where(df.time.str[:2] > '05', 'N', 'Y')

    # divide to each hour
    df['0am'] = np.where(df['time'].str[1] == '0', '0am', np.nan)
    df['1am'] = np.where(df['time'].str[1] == '1', '1am', np.nan)
    df['2am'] = np.where(df['time'].str[1] == '2', '2am', np.nan)
    df['3am'] = np.where(df['time'].str[1] == '3', '3am', np.nan)
    df['4am'] = np.where(df['time'].str[1] == '4', '4am', np.nan)

    # combine hour lable to 1 column to facilitate binning
    mine['hour'] = mine[['0am', '1am', '2am', '3am', '4am']].fillna("").sum(axis=1)

    # divide date to year, month and day, convert day to int
    df['year'] = df['date'].str[5:7]
    df['month'] = df['date'].str[5:7]
    df['day'] = df['date'].str[-2:]
    df['day'] = np.where(len(df['day']) > 1, df['day'].str[1], df['day'])

    return df


"tokenize tweets"
def tokenize(x):
    # remove stop words and convert selected words
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    avoid = ['i', 'twitter', 'com', 'don', 'didn', 'gon', 'hi',
             'pm', 'get', 'yeah', 'yes', 'tweet', 'isn', 'wow',
             've', 'th', 'doesn', 'lol', 'rt', 'fuck', 'hey',
             'we', 'day', 'us']
    convert = {'years': 'year', 'months': 'month', 'days': 'day',
               'hours': 'hour', 'thanks': 'thank', 'things': 'thing',
               'weeks': 'week', 'friends': 'friend'}

    word_list = []
    for (word, pos) in nltk.pos_tag(nltk.word_tokenize(txt)):

        if word in avoid or len(word) == 1 or word in stop_words:
            continue
        elif word in convert:
            word = convert[word]
        word_list.append(word)

    return word_list


"perform tweet text cleaning"
def transform(df):

    # remove all URLs, emojis, punctuations, numbers, convert words to lower cases
    df = df.apply(lambda x: re.split(r'https:\/\/.*', str(x))[0])
    df = df.apply(lambda x: re.sub(r'([^0-9A-Za-z \t])|(\w+:\/\/\S+)', " ", x))
    df = df.apply(lambda x: re.sub(r'[,.;@#?!&$%"\'\d+]+\ *', " ", x, flags=re.VERBOSE))
    df = df.apply(lambda x: x.lower())

    # tokenize words so that tweet strings are convered to word lists
    df['interpret'] = df['tweet'].apply(lambda x: tokenize(x))

    return df


"flatten list of sub word lists to a consolidated list of all elements"
def frequency(column):
    # platten transaction list
    transactions = column.values.tolist()
    return list(itertools.chain(*transactions))


"add time items to tweet word list to facilitate frequent itemset mining"
def frequent_item(df):

    # transform month and day to meaning text instead of numbers
    month = {'12': 'm_dec', '11': 'm_nov', '10': 'm_oct',
             '09': 'm_sept', '08': 'm_aug', '07': 'jul',
             '06': 'jun', '05': 'may', '04': 'apr', '03': 'mar',
             '02': 'feb', '01': 'jan'}

    df.month = df.month.apply(lambda x: x.replace(x, month[x]))
    df.day = df.day.apply(lambda x: x.replace(x, 'day_'+x))

    # add year, month, day, hour to word lists
    df['mine'] = df['mine'] + df['interpret']
    df['mine'] = df[['date', 'year', 'month', 'day', 'hour']].values.tolist()

    return df

#########################################################################################################
