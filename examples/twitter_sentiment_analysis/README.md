# Twitter Sentiment Analysis



This example showcases the use of `Forte` to build a retrieval-based chatbot and perform text 
analysis on the retrieved results. We use the dataset released as part of this paper 
[Target-Guided Open-Domain Conversation](https://arxiv.org/abs/1905.11553). The dataset consists 
of conversations between two entities A and B. We finetune a BERT model that helps retrieve a 
response for a context.

**Note**: All the commands below should be run from `examples/chatbot/` directory.

In this example, the user speaks in German and the bot extracts information stored in English. The 
bot finally translates the response to German. For text analysis, we run a *Semantic Role 
Labeler* (SRL) on the retrieved response to identify predicate mentions and arguments. Let us see a 
step-by-step guide on how to run this example.

## Using the example in inference mode

### Downloading the models

Before we run the chatbot, we need to download the models. 

- Download chatbot model

```bash
python download_models.py --model-name chatbot-bert
```

- Download the index

```bash
python download_models.py --model-name indexer
```

- Download the SRL model

```bash
python download_models.py --model-name srl
```