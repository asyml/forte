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

**Forte** is a toolkit for building Natural Language Processing pipelines, featuring
composable components, convenient data interfaces, and cross-task interaction. Forte designs
a universal data representation format for text, making it a
one-stop platform to assemble state-of-the-art NLP/ML technologies, ranging
from Information Retrieval, Natural Language Understanding to Natural Language Generation.

Forte was originally developed in CMU and is actively contributed
by [Petuum](https://petuum.com/)
in collaboration with other institutes. This project is part of
the [CASL Open Source](http://casl-project.ai/) family.

## Download and Installation

To install the released version from PyPI:

```bash
pip install forte
```

To install from source,

```bash
git clone https://github.com/asyml/forte.git
cd forte
pip install .
```

To install some forte adapter for some
existing [libraries](https://github.com/asyml/forte-wrappers#libraries-and-tools-supported):

```bash
git clone https://github.com/asyml/forte-wrappers.git
cd forte-wrappers
# Change spacy to other tools. Check here https://github.com/asyml/forte-wrappers#libraries-and-tools-supported for available tools.
pip install src/spacy
```

Some components or modules in forte may require some [extra requirements](https://github.com/asyml/forte/blob/master/setup.py#L45):

* `pip install forte[data_aug]`: Install packages required for [data augmentation modules](https://github.com/asyml/forte/tree/master/forte/processors/data_augment).
* `pip install forte[ir]`: Install packages required for [Information Retrieval Supports](https://github.com/asyml/forte/tree/master/forte/processors/ir/)
* `pip install forte[remote]`: Install packages required for pipeline serving functionalities, such as [Remote Processor](https://github.com/asyml/forte/processors/misc/remote_processor.py).
* `pip install forte[audio_ext]`: Install packages required for Forte Audio support, such as [Audio Reader](https://github.com/asyml/forte/blob/master/forte/data/readers/audio_reader.py).
* `pip install forte[stave]`: Install packages required for [Stave](https://github.com/asyml/forte/blob/master/forte/processors/stave/stave_processor.py) integration.
* `pip install forte[models]`: Install packages required for [ner training](https://github.com/asyml/forte/blob/master/forte/trainer/ner_trainer.py), [srl](https://github.com/asyml/forte/tree/master/forte/models/srl), [srl with new training system](https://github.com/asyml/forte/tree/master/forte/models/srl_new), and [srl_predictor](https://github.com/asyml/forte/tree/master/forte/processors/nlp/srl_predictor.py)
* `pip install forte[test]`: Install packages required for running [unit tests](https://github.com/asyml/forte/tree/master/tests).
* `pip install forte[wikipedia]`: Install packages required for reading [wikipedia datasets](https://github.com/asyml/forte/tree/master/forte/datasets/wikipedia).
* `pip install forte[nlp]`: Install packages required for additional NLP supports, such as [subword_tokenizer](https://github.com/asyml/forte/tree/master/forte/processors/nlp/subword_tokenizer.py) and [texar encoder](https://github.com/asyml/forte/tree/master/forte/processors/third_party/pretrained_encoder_processors.py)
* `pip install forte[extractor]`: Install packages required for extrator-based training system, [extractor](https://github.com/asyml/forte/blob/master/forte/data/extractors), [train_preprocessor](https://github.com/asyml/forte/tree/master/forte/train_preprocessor.py) and [tagging trainer](https://github.com/asyml/forte/tree/master/examples/tagging/tagging_trainer.py)



## Getting Started

* [Examples](./examples)
* [Documentation](https://asyml-forte.readthedocs.io/)
* Currently we are working on some
  interesting [tutorials](https://github.com/asyml/forte/wiki)


With Forte, it is extremely simple to build an integrated system that can search
documents, analyze, extract information and generate language all in one place.
This allows developers to fully utilize the strength of individual module,
combine the results from each step, and enables the system to make fully
informed decision at the end of the pipeline.

Forte not only makes it easy to integrate with arbitrary 3rd party tools (Check
out these [examples](./examples)!), but also brings technology to you by
offering a miscellaneous collection of deep learning modules via Texar, and a
convenient model-data interface for casting tasks to models.

## Library Example

A simple code example that runs Named Entity Recognizer from Spacy (required
installing forte spacy wrapper)

```python
from forte import Pipeline
from forte.data.readers import TerminalReader
from fortex.spacy import SpacyProcessor

for pack in Pipeline().set_reader(
        TerminalReader()
).add(
    SpacyProcessor(), {"processors": ["sentence", "ner"]}
).initialize().process_dataset():
    for sentence in pack.get("ft.onto.base_ontology.Sentence"):
        print("The sentence is: ", sentence.text)
        print("The entities are: ")
        for ent in pack.get("ft.onto.base_ontology.EntityMention", sentence):
            print(ent.text, ent.ner_type)

```


## Core Design Principles

The core design principle of Forte is the abstraction of NLP concepts and
machine learning models. It not only separates data, model and tasks but also
enables interactions between different components of the pipeline. Based on this
principle, we make Forte:

* **Composable**: Forte helps users to decompose a problem into *data*, *models*
  and *tasks*. The tasks can further be divided into sub-tasks. A complex use
  case can be solved by composing heterogeneous modules via straightforward
  python APIs or declarative configuration files. The components (e.g. models or
  tasks) in the pipeline can be flexibly swapped in and out, as long as the API
  contracts are matched. This approach greatly improves module reusability,
  enables fast development and enhances the flexibility of using libraries.

* **Generalizable and Extensible**: Forte not only generalizes well on a wide
  range of NLP tasks, but also extends easily to new tasks or new domains. In
  particular, Forte provides the *Ontology* system that helps users define types
  according to their specific tasks. Users can declaratively specify the type
  through simple JSON files and our Code Generation tool will automatically
  generate ready-to-use python files for your project. Check out our
  [Ontology Generation documentation](./docs/ontology_generation.md) for more
  details.

* **Universal Data Flow**: Forte enables a universal data flow that supports
  seamless data flow between different steps. Central to Forte's composable
  architecture, a transparent data flow facilitates flexible process
  interventions and simple pipeline management. Adaptive to generic data
  formats, Forte is positioned as a perfect tool for data inspection, component
  swapping and result sharing. This is particularly helpful during team
  collaborations!

-----------------
| ![forte_arch.jpg](https://raw.githubusercontent.com/asyml/forte/master/docs/_static/img/forte_arch.png) |
|:--:|
| *A high level Architecture of Forte showing how ontology and entries work with the pipeline.* |
-----------------
| ![forte_results.jpg](https://raw.githubusercontent.com/asyml/forte/master/docs/_static/img/forte_results.png) |
|:--:|
|*Forte stores results in data packs and use the ontology to represent task logic.* |
-----------------


## Contributing

If you are interested in making enhancement to Forte, please first go over
our [Code of Conduct](https://github.com/asyml/forte/blob/master/CODE_OF_CONDUCT.md)
and [Contribution Guideline](https://github.com/asyml/forte/blob/master/CONTRIBUTING.md)

## License

[Apache License 2.0](./LICENSE)

## Companies and Universities Supporting Forte

<p float="left">
   <img src="https://raw.githubusercontent.com/asyml/forte/master/docs/_static/img/Petuum.png" width="200" align="top">
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src="https://asyml.io/assets/institutions/cmu.png", width="200" align="top">
   <img src="https://www.ucsd.edu/_resources/img/logo_UCSD.png" width="200" align="top">
</p>
