<div align="center">
   <img src="https://raw.githubusercontent.com/asyml/forte/master/docs/_static/img/logo_h.png"><br><br>
</div>

-----------------

[![Build Status](https://github.com/asyml/forte/actions/workflows/main.yml/badge.svg)](https://github.com/asyml/forte/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/asyml/forte/branch/master/graph/badge.svg)](https://codecov.io/gh/asyml/forte)
[![Documentation Status](https://readthedocs.org/projects/asyml-forte/badge/?version=latest)](https://asyml-forte.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/asyml/forte/blob/master/LICENSE)
[![Chat](http://img.shields.io/badge/gitter.im-asyml/forte-blue.svg)](https://gitter.im/asyml/community)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Forte** is a toolkit for building Natural Language Processing pipelines, featuring composable components, convenient data interfaces, and cross-task interaction. Forte designs a universal data representation format for text, making it a one-stop platform to assemble state-of-the-art NLP/ML technologies, ranging from Information Retrieval, Natural Language Understanding to Natural Language Generation.

Forte was originally developed in CMU and is actively contributed by [Petuum](https://petuum.com/) in collaboration with other institutes. This project is part of the [CASL Open Source](http://casl-project.ai/) family.

## Download and Installation

To install the released version from PyPI:

```
pip install forte
```

To install from source,

```
git clone https://github.com/asyml/forte.git
cd forte
pip install .
```

To install some forte adapter for some existing [libraries](https://github.com/asyml/forte-wrappers#libraries-and-tools-supported):

```
git clone https://github.com/asyml/forte-wrappers.git
cd forte-wrappers
# Change spacy to other tools. Check here https://github.com/asyml/forte-wrappers#libraries-and-tools-supported for available tools.
pip install src/spacy
```

Some components or modules in forte may require some [extra requirements](https://github.com/asyml/forte/blob/master/setup.py#L45):

* ```pip install forte[ner]```: Install packages required for [ner_trainer](https://github.com/asyml/forte/blob/master/forte/trainer/ner_trainer.py)
* ```pip install forte[test]```: Install packages required for running [unit tests](https://github.com/asyml/forte/tree/master/tests).
* ```pip install forte[example]```: Install packages required for running [forte examples](https://github.com/asyml/forte/tree/master/examples).
* ```pip install forte[wikipedia]```: Install packages required for reading [wikipedia datasets](https://github.com/asyml/forte/tree/master/forte/datasets/wikipedia).
* ```pip install forte[augment]```: Install packages required for [data augmentation module](https://github.com/asyml/forte/tree/master/forte/processors/data_augment).
* ```pip install forte[stave]```: Install packages required for [StaveProcessor](https://github.com/asyml/forte/blob/master/forte/processors/stave/stave_processor.py).
* ```pip install forte[audio_ext]```: Install packages required for [AudioReader](https://github.com/asyml/forte/blob/master/forte/data/readers/audio_reader.py).

## Getting Started

* [Examples](./examples)
* [Documentation](https://asyml-forte.readthedocs.io/)
* Currently we are working on some interesting [tutorials](https://github.com/asyml/forte/wiki)


With Forte, it is extremely simple to build an integrated system that can search
documents, analyze, extract information and generate language all in one place.
This allows developers to fully utilize the strength of individual module, combine the results from each step, and enables the system to make fully informed decision at the end of the pipeline.

Forte not only makes it easy to integrate with arbitrary 3rd party tools (Check out these [examples](./examples)!), but also brings technology to you by offering a miscellaneous collection of deep learning modules via Texar, and a convenient model-data interface for casting tasks to models.

## Quick Start Guide
Consider a case we want to get words in the sentence and also extract entities from the sentence. We can write two processors for the two separate tasks.
First, we imports all required libraries.
```
from forte import Pipeline
from forte.processors.base import PackProcessor
from forte.data.data_pack import DataPack
from dataclasses import dataclass
from forte.data.ontology.top import Generics
from typing import Optional, Dict, Any
from forte.data.readers import TerminalReader
from fortex.spacy import SpacyProcessor
import re
```
In forte, we use data entry with different entry types to represent data.
The following class is an example of data entry storing `value`. New user
don't need to pay too much attention about it and just need to use it as it is.
```
@dataclass
class Word(Generics):
    """A dummy generic type for words."""
    value: Optional[str] = None
    def __init__(self, pack, value):
        super().__init__(pack)
        self.value = value
```
Next we can write a simple customized processor for the first task, split sentences into words.
There are two steps for the sentence processing.
First, we need to strip all punctuations as splitted words shouldn't contain them.
Second, we need to split stripped sentences into words and write words into the data pack.
```
class WordSplitPackProcessor(PackProcessor):
    """A processor that removes punctuations in sentences in the data pack
    and split sentences into words and write words into the data pack.
    """
    def __init__(self):
        super().__init__()

    def initialize(self, resources, configs):
        super().initialize(resources, configs)

    def _process(self, input_pack: DataPack):
        # write a data type Word into the data pack
        for sentence in pack.get("ft.onto.base_ontology.Sentence"):
            # first step: strip all punctuations
            words = re.sub(r'[^\w\s]','',sentence.text).split(" ")
            # second step: split stripped sentences into words and write
            # words into the data pack
            for w in words:
                Word(pack=input_pack, value=w)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        return {"test": "test, successor"}
```
Finally, we set up pipeline and add all pipeline components into it, and process
input read from the terminal.
```
pipeline: Pipeline = Pipeline[DataPack]()
# add a reader that reads input by prompting user in the terminal
pipeline.set_reader(TerminalReader())
# NOTE: we can add multiple processors and they process the input sequentially
#       without a conflict.
# add a SpacyProcessor from third party library that extract entity mentions
pipeline.add(SpacyProcessor(), {"processors": ["sentence", "ner"]})
# add the customized processor that split sentences into words
pipeline.add(WordSplitPackProcessor())
for pack in pipeline.initialize().process_dataset():
    for sentence in pack.get("ft.onto.base_ontology.Sentence"):
        print("The sentence is: ", sentence.text)
        print("The entities are: ")
        for ent in pack.get("ft.onto.base_ontology.EntityMention", sentence):
            print(ent.text, ent.ner_type)
    print("Customized WordSplitPackProcessor results: ")
    for token in pack.get(Word):
        print(token, end = " ")
    print()
```


-----------------
| ![forte_arch.jpg](https://raw.githubusercontent.com/asyml/forte/master/docs/_static/img/forte_arch.png) ||:--:| | *A high level Architecture of Forte showing how ontology and entries work with the pipeline.* |
-----------------
| ![forte_results.jpg](https://raw.githubusercontent.com/asyml/forte/master/docs/_static/img/forte_results.png) ||:--:||*Forte stores results in data packs and use the ontology to represent task logic.* |
-----------------


## Contributing

If you are interested in making enhancement to Forte, please first go over our [Code of Conduct](https://github.com/asyml/forte/blob/master/CODE_OF_CONDUCT.md) and [Contribution Guideline](https://github.com/asyml/forte/blob/master/CONTRIBUTING.md)

## License

[Apache License 2.0](./LICENSE)

## Companies and Universities Supporting Forte

<p float="left">
   <img src="https://raw.githubusercontent.com/asyml/forte/master/docs/_static/img/Petuum.png" width="200" align="top">
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src="https://asyml.io/assets/institutions/cmu.png", width="200" align="top">
   <img src="https://www.ucsd.edu/_resources/img/logo_UCSD.png" width="200" align="top">
</p>
