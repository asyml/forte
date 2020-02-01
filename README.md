<div align="center">
   <img src="./docs/_static/img/logo_h.png"><br><br>
</div>

-----------------

[![Build Status](https://travis-ci.org/asyml/forte.svg?branch=master)](https://travis-ci.org/asyml/forte)
[![Documentation Status](https://readthedocs.org/projects/asyml-forte/badge/?version=latest)](https://asyml-forte.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/asyml/forte/blob/master/LICENSE)


**Forte** is a flexible composable system designed for Text processing, providing integrated 
architecture support for a wide spectrum of tasks, from Information Retrieval to tasks in Natural 
Language Processing (including text analysis and language generation). Empowered by principled 
abstraction and design principles, Forte provides a platform to gather cutting-edge NLP and ML 
technologies in a composable manner. We will demonstrate that such abstraction enables better data, 
model and task separation, but provides stronger inter-task interactions at the same time. 
In addition, Forte provides strong support of deep learning and general machine learning. With 
these features, Forte help users to efficiently build customized modules and easily incorporate 
existing technologies to solve complex text processing problems.


## Core Design Principles

* **Composing via Contracts**: Forte has adopted a highly modularized structure in order to 
decompose  data, models and tasks, as well as separate the tasks into sub-steps. A complex use 
case can be solved by composing heterogeneous modules via straightforward python APIs or 
declarative configuration files. The components (e.g. models or tasks) in the pipeline can be 
flexibly swapped in and out, as long as the API contracts are matched. The approach greatly 
improves module reusability, enables fast development and makes the library flexible for user need.
   
* **Generalization for Extensibility**: Forte promotes generalization to support not only a wide 
range of NLP tasks, but also extensible for new tasks or new domains. Task logic is reflected 
through the Ontology system, that defines a general structure to represent NLP data structures. 
The ontology system enables appropriate APIs at varying abstraction level, and can be flexibly 
extended so specific domain knowledge can be injected.

* **Transparent and Universal**: Forte supports a full pipeline of many different stages in text 
processing. This is achieved by designing a universal data format to support seamless data flow in 
between the steps. Forte advocates a transparent data flow to facilitate flexible process 
intervention and simple pipeline control. Combined with the general data format, Forte makes a 
perfect tool for data inspection, component swapping and result sharing.

<figure class="image">
   <img src="./docs/_static/img/forte_arch.png"><br><br>
   <figcaption>A high level Architecture of Forte showing how ontology and entries work with the 
   pipeline.</figcaption>
</figure>

<figure class="image">
   <img src="./docs/_static/img/forte_results.png"><br><br>
   <figcaption>Forte stores results in data packs and use the ontology to represent task logic
   </figcaption>
</figure>

### Library API example

A code example that builds and runs a Chatbot pipeline

```python
import yaml

from termcolor import colored
from texar.torch import HParams

from forte.data.readers import MultiPackTerminalReader
from forte.common.resources import Resources
from forte.pipeline import Pipeline
from forte.processors import MicrosoftBingTranslator, BertBasedQueryCreator, \
    SRLPredictor
from forte.processors.search_processor import SearchProcessor
from forte.data.selector import NameMatchSelector
from forte.processors.nltk_processors import \
    (NLTKSentenceSegmenter, NLTKWordTokenizer, NLTKPOSTagger)

from ft.onto.base_ontology import PredicateLink, Sentence


config = yaml.safe_load(open("config.yml", "r"))
config = HParams(config, default_hparams=None)

resource = Resources()
query_pipeline = Pipeline(resource=resource)
query_pipeline.set_reader(
    reader=MultiPackTerminalReader(), config=config.reader)

query_pipeline.add_processor(
    processor=MicrosoftBingTranslator(), config=config.translator)
query_pipeline.add_processor(
    processor=BertBasedQueryCreator(), config=config.query_creator)
query_pipeline.add_processor(
    processor=SearchProcessor(), config=config.indexer)
query_pipeline.add_processor(
    processor=NLTKSentenceSegmenter(),
    selector=NameMatchSelector(
        select_name=config.indexer.response_pack_name[0]))
query_pipeline.add_processor(
    processor=NLTKWordTokenizer(),
    selector=NameMatchSelector(
        select_name=config.indexer.response_pack_name[0]))
query_pipeline.add_processor(
    processor=NLTKPOSTagger(),
    selector=NameMatchSelector(
        select_name=config.indexer.response_pack_name[0]))
query_pipeline.add_processor(
    processor=SRLPredictor(), config=config.SRL,
    selector=NameMatchSelector(
        select_name=config.indexer.response_pack_name[0]))
query_pipeline.add_processor(
    processor=MicrosoftBingTranslator(), config=config.back_translator)

query_pipeline.initialize()

for m_pack in query_pipeline.process_dataset():

    english_pack = m_pack.get_pack("pack")
    print(colored("English Translation of the query: ", "green"),
          english_pack.text, "\n")
    pack = m_pack.get_pack(config.indexer.response_pack_name[0])
    print(colored("Retrieved Document", "green"), pack.text, "\n")
    print(colored("German Translation", "green"),
          m_pack.get_pack("response").text, "\n")
          
    for sentence in pack.get(Sentence):
        sent_text = sentence.text
        print(colored("Sentence:", 'red'), sent_text, "\n")
    
        print(colored("Semantic role labels:", 'red'))
        for link in pack.get(PredicateLink, sentence):
            parent = link.get_parent()
            child = link.get_child()
            print(f"  - \"{child.text}\" is role {link.arg_type} of "
                  f"predicate \"{parent.text}\"")
        print()
    
        input(colored("Press ENTER to continue...\n", 'green'))

```

Many more examples are available [here](./examples).

### Download and Installation

Download the repository through

```bash
git clone https://github.com/asyml/forte.git
```

After `cd` into `forte`, you can install it through

```bash
pip install .
```

### Getting started

* [Examples](./examples)
* [Documentation](https://asyml-forte.readthedocs.io/)

### Reference

### License

[Apache License 2.0](./LICENSE)