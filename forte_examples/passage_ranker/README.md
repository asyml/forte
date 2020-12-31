# Passage Ranker
This example showcases the use of `Forte` to build a complete two stage IR pipeline, namely the retrieval and the re-ranking. The retrieval stage typically focuses on obtaining as many relevant documents as possible (high recall) in low cost, while the re-ranking stage aims to produce a high-quality ranked list of top `n` desired number of results (high precision). We demonstrate the pipeline through the [MS-MARCO Passage Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking) use-case. We use the `Full Ranking` as the Ranking Style, which includes retrieval and re-ranking (as opposed to `Re-Ranking` where the task is to use BM25 retrieved documents and rerank them). We use the TREC 2019 version of `Passage Ranking` dataset which can be found [here](https://microsoft.github.io/TREC-2019-Deep-Learning/).

> All the examples are run from the directory `example/passage_ranker`

## Inference Mode

Download MS-MARCO Passage Ranking dataset
```
python download_data.py
```

Index passages present in the corpus file `data/collectionandqueries/collections.tsv` using Elasticsearch. Elasticsearch configuration can be changed in `config.yml`
```
python create_index.py
```

Download pre-trained BERT reranker (`bert-base-uncased` or `bert-large-uncased`) made available by [Nogueira et al](https://github.com/nyu-dl/dl4marco-bert)[1]. The argument `pretrained_model_name` also accepts a user provided URL that points to a pretrained BERT model.
```
python download_model.py --pretrained_model_name bert-base-uncased
```

To query the corpus using command line, run - 
```
python indexer_reranker_eval_pipeline.py
```

To run a batch of queries and obtain evaluation scores -
```
python indexer_reranker_eval_pipeline.py --input_file data/collectionandqueries/query_doc_id.tsv
```

> All the above scripts use `config.yml` by default. The config file can be modified accordingly. The command line arguments, whenever available, take precedence over `config.yml`.

## Training Mode
Coming soon.

[1] Nogueira, Rodrigo, and Kyunghyun Cho. "Passage Re-ranking with BERT." arXiv preprint arXiv:1901.04085 (2019).
