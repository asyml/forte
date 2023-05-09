# Sentence Sentiment Classifier

This is a Sentence Sentiment Classifier as an example of Classification Task

The example contains two classifier, Conv Classifier and Bert Classifier

The example shows:
  * Training and predicting pipeline using Forte 
  * How to write a reader for [IMDB dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
  * How to set configuration in Train Preprocessor
  * How to switch Classifier from CNN to Bert
  * How to add data augmentation
  
 # Usage
Use the following to train the network:
```
python main_train.py
```
Use the following to train the network:
```
python main_predict.py
```

If you want to switch model from CNN to Bert, set the "model" field in [config_model.yml](./config_model.yml)

You can also define your own classifier or network in another python file like [cnn.py](./cnn.py) here

Define you utility function in [util.py](./util.py) 