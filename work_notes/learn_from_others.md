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
    ```
    nlp = nlp_pipeline(config)
    
    # Input.
    for pack in nlp.from_corpus("corpus_path", "corpus_type").process_next():
        pack
    #or
        pack = nlp('blah blah'):
       
    # Option 1
    tokens_text = [token.text for token in pack.get(Token)]
    
    for sentence in pack.get(Sentence):
        for token in sentence.get(Token);
            print(token.text)            
        for entity_mention in sentence.get(EntityMention):
            print(entity_mention.type)
            coref_group = entity_mention.get(EntityCoreference)
            
    # Option 2
    for sentence in pack.get(Sentence):
        for token in pack.get(sentence, Token):
            print(token.text)
        for entity_mention in pack.get(sentence, EntityMention):
            print(entity_mention.type)
            coref_group = pack.get(entity_mention, EntityCoreference)                        
    ```

# Cons
1. Problem: Assumes token to be the lower level representation (For example, 
    cannot support BPE)
1. Data is highly packed hence difficult to manipulate from outside.

