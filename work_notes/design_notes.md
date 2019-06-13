Design Notes
===

# The interchange format
***Efficiency Concerns***
   
1. Meta data such as components takes a large portion in
   the json file.
1. Python is not a very fast language for this sort of things,
   we need to make sure the format is lanuage-independent, allow
   possible future extension.

***Readability Concerns***
1. The core of the format is simple, but it gets large with all
   the meta data. They might look daunting to people.
    
---

# Interface to learning modules

***Usability Concerns***
1. We have implemented the token based indexing method, which is
the major way to think of NLP data.

---
