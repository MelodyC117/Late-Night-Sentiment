#########################################################################################################
""" This script was solely used for DATA VISUALIZATION to plot word frequency
    and sentiment distribution;
    Plots include -
        x. Word Cloud with WordCloud
        x. Topic modeling with
        x. Sentiment Distribution with Matplotlib
        x. Additional plots utilize Excel
"""

import pandas as pd
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns

################################################# WORD CLOUD ############################################

"generate word clouds based on cleaned word list"
def word_cloud(fontsize, num_words, transaction):

    wordcloud = WordCloud(max_font_size=fontsize, max_words=num_words,
                          background_color="white").generate(" ".join(transaction))

    plt.figure(figsize=(15, 15));
    plt.imshow(wordcloud, interpolation="bilinear");
    plt.axis("off");
    plt.show();


################################################## pyLDAvis #############################################


"generate pyLDAvis visualization to desmontrate each clustered topic"
def get_pyLDAvis(lda_object, corpus, id2word):

    # shows interactive bar chart by clusters and distance map for each cluster
    pyLDAvis.enable_notebook()
    graph = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

    return graph


########################################## SENTIMENT DISTRIBUTION ########################################


"helps visualize distribution of sentiment scores
"based on [mean, min, max] values and on hours of the night"
def get_distribution(series, color, alpha, title, xlab, ylab):
    plt.figure(figsize=(10, 8))

    # additional customization features are added outside the function
    plt.hist(series, color=color, alpha=alpha);
    plt.title(title);
    plt.xlabel(xlab);
    plt.ylabel(ylab);

#########################################################################################################
