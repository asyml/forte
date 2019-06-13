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
        
1. Python is not a very fast language for this sort of things,
we need to make sure the format is language-independent, allow
possible future extension.
    1. There are currently some fields unique to json_pickle, but they are just
    indexing meta which do not affect the semantics.

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
