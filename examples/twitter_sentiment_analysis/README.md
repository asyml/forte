# Twitter Sentiment Analysis

This example show the use of `Forte` to perform sentiment
analysis on the user's retrieved tweets, based on [Tweepy](https://docs.tweepy.org/en/latest/index.html), [Twitter API](https://developer.twitter.com/en/products/twitter-api) and 
[Vader (Valence Aware Dictionary and Sentiment Reasoner)](https://github.com/cjhutto/vaderSentiment).
 

> **Note**: To run this example, you need to have a Twitter account and apply for Developer Access, 
then create an application. It will generate the API credentials that you will need use to access Twitter from Python.
You should put the credentials at `api_credential.yml` first to make the pipeline work. 
You could refer to 
https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api
 for more information.


## How to run the pipeline

First, you need to create a virtual environment, then in command line:

`cd twitter_sentiment_analysis`

`pip install -r requirements.txt`


We can run the pipeline by run

`python pipeline.py`

Then you can input your search query in terminal to get the tweets and sentiment scores.

You can also refer to Twitter's official documentation 
https://developer.twitter.com/en/docs/twitter-api/tweets/search/integrate/build-a-query
for customized query.
