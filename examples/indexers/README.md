# Retrieval-based Chatbot

This example showcases the use of `Forte` to build a retrieval-based chatbot and perform text 
analysis on the retrieved results. We use IMDB dataset released from Stanford as a database to 
generate bot's response.

Note: All the commands below should be run from `examples/indexers/` directory.

In this example, the user speaks in German and the bot extracts information stored in English. The 
bot finally translates the response to German. Let us see a step-by-step guide on how to use run 
this example.

-  Run the `download_imdb_data.sh` script located in `scripts/` folder to download and extract 
IMDB dataset.

- We process the dataset so that it can be easily used to create an index. Run the following 
command

  ```bash
  python prepare_data.py
  ```

- To start the chatbot, run the command

  ```bash
  python indexer_example.py
  ```

  Note: In the above example, we first create the index if one does not exist already.

  The user is prompted to start a conversation by entering a query in German. The bot then 
  translates the query to English, fetches the most relevant example from the database, runs 
  Semantic Role Labeling model on the result to identify predicates and arguments. It finally 
  translates and returns the response in German. The bot continues to hold a conversation with the 
  user until termination (`Ctrl + D`). The bot currently uses the context of the last turn to 
  enrich the response of the current turn i.e the bot uses the utterances from the last turn to 
  improve search results in the current turn.


### Temporary settings


The machine translation processor currently uses Bing APIs to perform the translation Please set the
following environment variable to use the processor

```
export MICROSOFT_API_KEY=2699c6fb32684483ae050b45f2af3de9
```