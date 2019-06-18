What are the good features in other NLP pipeline?
===
General
---
1. Supports quick corpus reading
    - SpaCY reads CoNLL.
    - torch-text supports imdb, babi, common language modeling corpus
    - We should support the popular datasets (which ones?)
1. Supports various pre-processing
    - Noise text cleaning (HTML etc.)
1. Strong models
    - AllenNLP on SRL and Coref
    - fasttext on classification
1. Various embedding support
    - We should definitely support ELMO and BERT
  

SpaCY
---

# Pros
1. Fast
    1. C level support on data structures
1. Multi-language support (trained with Wikipedia sliver standard)
1. Easy to use
    ```
    from spacy.lang.en import English
    nlp = English()
    tokens = nlp(u"Some\nspaces  and\ttab characters")
    tokens_text = [t.text for t in tokens]        
    ```
    Our method is slightly indirect comparing to this, but we may be able to 
    achieve this via some syntax sugar.

# Cons
1. Problem: Assumes token to be the lower level representation (For example, 
    cannot support BPE)
1. Data is highly packed hence difficult to manipulate from outside.

