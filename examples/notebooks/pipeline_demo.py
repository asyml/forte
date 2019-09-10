#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from termcolor import colored

from forte.data import conll03_ontology as Ont
from nlp.forte.pipeline import Pipeline

# # Creates the pipeline here

# In[ ]:


pl = Pipeline()

pl.init_from_config_path("sample_config.yml")
# # The pipeline can wrap any external tools, for example, we are wrapping some NLTK tools.

# # Our pipeline is ready, now let's try out some text snippets.

# In[ ]:

search_engine_text = "A Scottish firm is looking to attract web surfers with a search engine that reads out results." \
                     " Called Speegle, it has the look and feel of a normal search engine, with the added feature of being able to read" \
                     " out the results. Scottish speech technology firm CEC Systems launched the site in November. But experts have" \
                     " questioned whether talking search engines are of any real benefit to people with visual impairments. The" \
                     " Edinburgh-based firm CEC has married speech technology with ever-popular internet search. The ability to search is" \
                     " becoming increasingly crucial to surfers baffled by the huge amount of information available on the web."

win_medal_text = "British hurdler Sarah Claxton is confident she can win her first major medal at next " \
                 "month's European Indoor Championships in Madrid."

# # Process this snippet with one simple command.

# In[ ]:


pack = pl.process_one(win_medal_text)

# # Now all the results are ready.
# ## We have added the results as "entries" into our data.
# ## Let's take a look at the sentences.

# In[ ]:


for sentence in pack.get(Ont.Sentence):
    sent_text = sentence.text
    print(colored("Sentence:", 'red'), sent_text, "\n")

# # We can access more fine-grained data in the sentences using our magical "get" function.
# ## Let's get all the tokens in the first sentence and print out their Part-of-Speech value.

# In[ ]:


for sentence in pack.get(Ont.Sentence):
    tokens = [(token.text, token.pos_tag) for token in
              pack.get(Ont.Token, sentence)]
    print(colored("Tokens:", 'red'), tokens, "\n")
    break

# ## Similarly, we can get all the named entities in the sentences, let's look at their types.

# In[ ]:


for sentence in pack.get(Ont.Sentence):
    for entity in pack.get(Ont.EntityMention, sentence):
        print(colored("EntityMention:", 'red'),
              entity.text,
              'has type',
              colored(entity.ner_type, 'blue'), "\n")

# ## With this simple "get" function we can do a lot more. Let's see how one can play with semantic role labeling and NER at the same time.

# In[ ]:

#
# for sentence in pack.get(Ont.Sentence):
#     print(colored("Semantic role labels:", 'red'))
#     # Here we can get all the links within this sentence.
#     for link in pack.get(Ont.PredicateLink, sentence):
#         parent = link.get_parent()
#         child = link.get_child()
#         print(f"  - \"{child.text}\" is role {link.arg_type} of predicate \"{parent.text}\"")
#         entities = [entity.text for entity in pack.get(Ont.EntityMention, child)]
#         print("      Entities in predicate argument:", entities, "\n")
#     print()


# In[ ]:


for sentence in pack.get(Ont.Sentence):
    for entity in pack.get(Ont.EntityMention, sentence):
        print(f"Entity: {entity.text}")
        for token in pack.get(Ont.Token, entity):
            print(f"Has token {token.text}")

# In[ ]:
