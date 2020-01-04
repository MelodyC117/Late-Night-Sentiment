#########################################################################################################
""" This script was solely used for DATA COLLECTION from twitter
    Tweets were collected first from [TwitterCrapper],
    then used the unique user IDs to collect addition tweets from [Twint] """

"1). Used TwitterCrapper through the command line to generated 1M tweets"

     "COMMAND LINE
     "[twitterscraper ,-l 1000000 -bd 2017-01-01 -ed 2019-12-30 --p 30 -o tweets.json]
     "Indicate keyword = , (empty query not allowed);
     "limit = 1M;
     "dates: [1/1/17 - 12/6/19];
     "paralell poolsize: 30;
     "output file: tweets.json"
     "data type: JSON;"

"2). Read the above JSON file into pandas's DataFrame and select 20,000 unique IDs;

import pandas as pd
import numpy as np

df = pd.read_json("tweets.json")
df = df[['text', 'timestamp','user_id', 'username']]

uid = pd.DataFrame(df['user_id'].unique())
select = np.random.choice(uid, 20000)

select.to_csv("ids.csv", index=False)

"3). Read the IDs into pandas's DataFrame and scrape 50 tweets for each ID; "

import twint
import pandas as pd

df = pd.read_csv("ids.csv")
uid = df['0']
uid = uid.tolist()

c = twint.Config()
fields = ["tweet","user_id", "date","time"]

for i in uid:
    c.User_id = i
    # for each tweet take the 4 fields as stated
    c.Custom["tweet"] = fields
    # restrict language to English only
    c.Lang = "en"
    # 50 most recent tweets are scrapped
    c.Limit = 50
    c.Output = "raw_output.csv"
    c.Store_csv = True
    # ignore error resulted from not enough tweets
    try:
        twint.run.Search(c)
    except:
        continue

#########################################################################################################
