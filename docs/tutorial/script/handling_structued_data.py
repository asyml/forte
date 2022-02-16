#!/usr/bin/env python
# coding: utf-8

# # Handling Structured Data
#
#
# ## Retrieve data
# `DataPack.get()` and `DataPack.get_data()` are methods commonly used to retrieve data from a `DataPack`.
# Let's start with introducing `DataPack.get()` which returns a generator that generate requested data __instance__.
#
# We can set up the `data_pack` using the following code.

import os

from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.utils import utils
from ft.onto.base_ontology import (
    Token,
    Sentence,
    Document,
    AudioAnnotation,
    AudioUtterance,
)
from forte.data.ontology import Annotation
from forte.data.readers import OntonotesReader, AudioReader
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline

data_path = os.path.abspath(
    os.path.join("../../data_samples", "ontonotes/one_file")
)
pipeline: Pipeline = Pipeline()
pipeline.set_reader(OntonotesReader())
pipeline.initialize()
data_pack: DataPack = pipeline.process_one(data_path)


# The following code explains how to retrieve data instance and access data fields in it.

for doc_idx, instance in enumerate(data_pack.get(Document)):
    print(doc_idx, "document instance:  ", instance)
    print(doc_idx, "document text:  ", instance.text)


# As we can see, we can get data instance from the generator returned by `data_pack.get(Document)`. And we can get the document text by `instance.text`.
#
# By contrast, `DataPack.get_data()` returns a generator that generates __dictionaries__ containing data requested, and each dictionary has a scope that covers __certain range of data__  in the `DataPack`.
#
# To understand this, let's consider a dummy case.
# Given that there is a document in the `DataPack` instance `data_pack`, we want to get the full document in `data_pack`.

#
# Then we can run the following code to get the full document.

for doc_idx, doc_d in enumerate(data_pack.get_data(context_type=Document)):
    print(doc_idx, ":  ", doc_d["context"])


# As we can see, the generator generates a dictionary each iteration (in this dummy case we only have one iteration) and the document data is retrieved by dictionary key `'context'`.
#
# To better understand this, let's consider a more concrete case. Since the document containing two sentences, suppose we want to retrieve text data sentence by sentence for a linguistic analysis task. In other words, we expect two dictionaries in the generator and each dictionary stores a sentence.
#
# We can get each sentence by the following code
#
#

data_generator = data_pack.get_data(context_type=Sentence)
for sent_idx, sent_d in enumerate(data_generator):
    print(sent_idx, sent_d["context"])


# As we can see, we get the two sentences by two iterations.
#
# So far, we introduce two examples to explain the first parameter `context_type` which controls the granularity of the data context. Depending on the task, we can generate data of different granularities. We assigned `context_type` from `Document` to `Sentence` for sentence tasks, and we can even further change it to `Token` for token tasks.
#
# Suppose we don't want to analyze the first sentence in the `data_pack`, there is `skip_k` parameter that skips k data of `context_type` and starts generating data from (k+1)th instance. In this case, we want to start generating from the second instance so we set `skip_k` to 1 to skip the first instance.

data_generator = data_pack.get_data(context_type=Sentence, skip_k=1)
for sent_idx, sent_d in enumerate(data_generator):
    print(sent_idx, sent_d["context"])


# Up until now, we have introduced three "data types", `Document`, `Sentence`, and `Token`. They are three common data entries for text analysis.
#
# They are also subclasses of `Annotation` which is a parent class for text data entries and can record text span which is the range of data that we have been explaining. However, such a retrieval is usually not flexible enough for a real task.
#
# Suppose we want to do part-of-speech tagging for each sentence, it means we need to tag `Token` pos within each sentence. Therefore, we need data entries of `Token` and `Sentence`. Moreover, we want to analyze POS sentence by sentence and `Token` data entries and its POS are better nested in retrieved `Sentence` data. Same as before, we should set `context_type` to be `Sentence`. Moreover, we introduce parameter `request` which supports retrieval of `Token` and its POS within the scope of `Sentence` context type.
#
# See the example below for how to set `requests`, and for simplicity we still skip the first sentence.

requests = {
    Token: ["pos"],
}
data_generator = data_pack.get_data(
    context_type=Sentence, request=requests, skip_k=1
)
for sent_idx, sent_d in enumerate(data_generator):
    print(sent_idx, sent_d["context"])
    print(sent_d["Token"]["pos"])
    print("Token list length:", len(sent_d["Token"]["text"]))
    print("POS list length:", len(sent_d["Token"]["pos"]))


# From the example, we can see `requests` is dictionary where keys are data entries of `Annotation` type and values are requested data entry attributes. And the retrieved data dictionary `sent_d` now has key `Token`, and `sent_d['Token']` is a dictionary has a key `pos`. It's exactly data entries what we requested.
#
#
# Moreover, we should pay attention to the range of `Token` data, values of `sent_d['Token']` is a list of data which are all within one sentence, and lists' length are all the same since each list item is one `Token`'s data.
#
#
# See example below to see the dissembled data and their correspondence.
#

data_generator = data_pack.get_data(
    context_type=Sentence, request=requests, skip_k=1
)
for sent_idx, sent_d in enumerate(data_generator):
    print(sent_idx, sent_d["context"])
    for token_txt, token_pos in zip(
        sent_d["Token"]["text"], sent_d["Token"]["pos"]
    ):
        print(token_txt, token_pos)


# intialize a token data dictionary
data_generator = data_pack.get_data(context_type=Token, skip_k=1)
token_d = next(data_generator)

print(doc_d.keys())  # document data dictionary
print(sent_d.keys())  # sentence data dictionary
print(token_d.keys())  # token data dictionary


# As we check dictionary keys for document, sentence and token data returned by the `get_data` method, there are four data fields. Except for `Token` we requested earlier, all other three are returned by default.
#
# A natural question arises: do those data classes have a parent class that shares common attributes of `'context', 'offset', 'tid'`. The answer is positive. We have `Annotation` class that represent generic text data.
# * `context`: data within the context type scope.
# * `offset`: the first character of the text class index
# * `tid`: id of the text data instances.
#
# Below we will dive in the attributes of `Annotation` class.

#
# ### Annotation
#
# In forte, each annotation has an attribute `span` which represents begin and end of annotation-specific data of that particular annotation. For `Annotation` type, range means the begin index and end index of characters under `Annotation` type in the `text` payload of the `DataPack`.

#
#  For an `Token` instance which is a subtype of `Annotation`, its annotation-specific data is `text` and therefore range means the begin and end of characters of that `Token` instance. For an `Recording` instance which is a subtype of `AudioAnnotation`, its annotation-specific data is `audio` and there range means the begin and end index of that `Recording` instance.
#
#
#
# As we are extending forte's capabilities of dealing more modalities, we also have a parent class for audio data which is `AudioAnnotation`.

# ### AudioAnnotation
# Based on the idea of "range", in the example code, entry `AudioUtterance` will be searched in `DataPack.audio_annotations` and the requested data field `speaker` will be included in the generator's data.
#
# For `AudioAnnotation` type, range means the begin index and end index of sound sample under `AudioAnnotation` type in the `audio` payload of the `DataPack`.
#
# For example, if User wants to get data of `AudioAnnotation` from a `DataPack` instance `pack`. User can call the function like the code blow. It returns a generator that User can iterate over.
# `AudioAnnotation` is passed into the method as parameter `context_type`.

# ## Build Coverage Index
# `DataPack.get()` is commonly used to retrieve entries from a datapack. In some cases, we are only interested in getting entries from a specific range. `DataPack.get()` allows users to set `range_annotation` which controls the search area of the sub-types. If `DataPack.get()` is called frequently with queries related to the `range_annotation`, you may consider building the coverage index regarding the related entry types. Users can call `DataPack.build_coverage_for(context_type, covered_type)` in order to create a mapping between a pair of entry types and target entries that are covered in ranges specified by outer entries.
#
# For example, if you need to get all the `Token`s from some `Sentence`, you can write your code as:

# Iterate through all the sentences in the pack.
for sentence in data_pack.get(Sentence):
    # Take all tokens from a sentence
    token_entries = data_pack.get(entry_type=Token, range_annotation=sentence)


# However, the snippet above may become a bottleneck if you have a lot of `Sentence` and `Token` entries inside the datapack. To speed up this process, you can build a coverage index first:

# Build coverage index between `Token` and `Sentence`
data_pack.build_coverage_for(context_type=Sentence, covered_type=Token)


# This `DataPack.build_coverage_for(context_type, covered_type)` function is able to build a mapping from `context_type` to `covered_type`, allowing faster retrieval of inner entries covered by outer entries inside the datapack.
# We also provide a function called `DataPack.covers(context_entry, covered_entry)` for coverage checking. It returns `True` if the span of `covered_entry` is covered by the span of `context_entry`.
#
