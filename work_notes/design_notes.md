Design Notes
===

# The interchange format
***Efficiency Concerns***
   
1. Meta data such as components takes a large portion in
the json file.
    1. Idea: Store component as a separate meta-data like this:
        ```
        "components_records": {
            "ontonotes_reader": {
                # These spans are created by ontonotes_reader
                'creation': [0, 1, 2, 5, 10],
                'attributes': {
                    # Attributes here are set by ontonotes_reader
                    'lemma': [5]
                }
            }
            "magic_lemmatizer": {
                # It didn't actually created anything.
                'creation': [],
                'attributes': {
                    # Attributes here are set by it
                    'lemma': [0, 1, 2, 10]
                }
            }
        }

        ``` 
        This method stores a separate list of component records and can be 
        very expressive. They can also be simply dropped if we would like to
        pass on less metadata but only the core content.
        
        A main problem of this component record is that once we added or altered
         the order, then we need to update the record every time.
1. Add component record
    1. Once we added some records, then we need to add to the meta that these
    records are there (add_record function), if we let the user to do it then it
    is quite cumbersome. But if we automatically do it every time then it may be
    a performance concern. 
1. Python is not a very fast language for this sort of things,
we need to make sure the format is language-independent, allow
possible future extension.
    1. There are currently some fields unique to json_pickle, but they are just
    indexing meta which do not affect the semantics.
    
1. Supporting multi-doc processing in the pipeline.
    1. Adding another layer on top of Pack, called Box
    1. A data box is a container of multiple packs
    1. The box can have links and groups to support cross-document references
        - e.g. Link can be used to create alignment (for MT or summarization)
        - e.g. Group can be used to represent cross-document coreference
    1. Box should contain named tuple of Packs
        - We can add a "source" pack and a "target" pack, but also refer them
        via indices
    1. The pipeline should be able to handle box in the same way as packs.
        - Creates extra complexity in batching.
        - We can start by only allowing box-level batching (i.e.
         can not create batches of half-boxes)
    1. Now, which multi-doc examples should we include?


***Readability Concerns***
1. The core of the format is simple, but it gets large with all
   the meta data. They might look daunting to people.
   1. One reason is the full qualified name, but that's kinda inevitable.
   1. If we move the component names out, that will reduce some boilerplates.
    
---

# Interface to learning modules

***Usability Concerns***
1. We haven't implemented the token based indexing method, which is
the major way to think of NLP data.

---
