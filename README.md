<div align="center">
   <img src="https://raw.githubusercontent.com/asyml/forte/master/docs/_static/img/logo_h.png"><br><br>
</div>

-----------------
<p align="center">
   <a href="https://github.com/asyml/forte/actions/workflows/main.yml"><img src="https://github.com/asyml/forte/actions/workflows/main.yml/badge.svg" alt="build"></a>
   <a href="https://codecov.io/gh/asyml/forte"><img src="https://codecov.io/gh/asyml/forte/branch/master/graph/badge.svg" alt="test coverage"></a>
   <a href="https://asyml-forte.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/asyml-forte/badge/?version=latest" alt="documentation"></a>
   <a href="https://github.com/asyml/forte/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="apache license"></a>
   <a href="https://gitter.im/asyml/community"><img src="http://img.shields.io/badge/gitter.im-asyml/forte-blue.svg" alt="gitter"></a>
   <a href="https://github.com/psf/black"><img src="http://img.shields.io/badge/gitter.im-asyml/forte-blue.svg" alt="code style: black"></a>
</p>

<p align="center">
  <a href="#download-and-installation">Download</a> •
  <a href="#quick-start-guide">Quick Start</a> •
  <a href="#license">License</a> •
  <a href="#contributing">Contribution Guide</a> •
  <a href="https://aclanthology.org/2020.emnlp-demos.26/">Publication</a>
</p>

**Bring good software engineering to your ML solutions, starting from Data!** 

**Forte** introduces [Data Pack](https://asyml-forte.readthedocs.io/en/latest/notebook_tutorial/handling_structued_data.html), a standardized data structure for unstructured data, distilling 
good software engineering practices such as reusability, extensibility, and flexibility into 
ML solutions. 

![image](https://user-images.githubusercontent.com/1015991/164107272-593ef68f-7438-4f11-9b76-251435995943.png)

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

To install some forte adapter for some existing [libraries](https://github.com/asyml/forte-wrappers#libraries-and-tools-supported):

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
* `pip install forte[extractor]`: Install packages required for extractor-based training system, [extractor](https://github.com/asyml/forte/blob/master/forte/data/extractors), [train_preprocessor](https://github.com/asyml/forte/tree/master/forte/train_preprocessor.py) and [tagging trainer](https://github.com/asyml/forte/tree/master/examples/tagging/tagging_trainer.py)


## Quick Start Guide
Writing NLP pipelines with Forte is easy. The following example creates a simple pipeline that analyzes the sentences, tokens, and named entities from a piece of text.

First, we imports all required libraries.
```python
from forte import Pipeline
from forte.processors.base import PackProcessor
from forte.data.data_pack import DataPack
from forte.data.readers import TerminalReader
from fortex.spacy import SpacyProcessor
from ft.onto.base_ontology import Token
from forte.processors.misc import WhiteSpaceTokenizer
import nltk
```
Next, we can write a simple customized processor for the first task, analyzing POS tags to tokens.
There are two steps for sentence processing.
First, we need to strip all punctuation as split words should not contain them.
Second, we need to split stripped sentences into words and write words into the data pack.
```python
class NLTKPOSTagger(PackProcessor):
    r"""A wrapper of NLTK pos tagger."""

    def initialize(self, resources, configs):
        super().initialize(resources, configs)
        # download tagger using average perceptron neural network
        nltk.download("averaged_perceptron_tagger")

    def __init__(self):
        super().__init__()

    def _process(self, input_pack: DataPack):
        # get a list of token data entries from `input_pack`
        # using `DataPack.get()`` method
        token_entries = list(
            input_pack.get(Token)
        )
        # get a list of token data entries text
        token_texts = [token.text for token in token_entries]
        # use nltk pos tagging module to tag token texts
        taggings = nltk.pos_tag(token_texts)
        # assign nltk taggings to token data entry attributes
        for token, tag in zip(token_entries, taggings):
            # tag is a tuple: (token text, tag)
            token.pos = tag[1]
```
Finally, we set up pipeline and add all pipeline components into it, and process the input read from the terminal.
```python
pipeline: Pipeline = Pipeline[DataPack]()
# set a reader that reads input by prompting user in the terminal
pipeline.set_reader(TerminalReader())
# add the first processor: SpacyProcessor from third party library that extract entity mentions
pipeline.add(SpacyProcessor(), {"processors": ["sentence", "ner"]})
# add the second processor: the tokenizer that tokenize a sentence into tokens
pipeline.add(WhiteSpaceTokenizer())
# add the third processor: a cutomized NLTK POS tagger that tags token texts
pipeline.add(NLTKPOSTagger())
for pack in pipeline.initialize().process_dataset():
    for sentence in pack.get("ft.onto.base_ontology.Sentence"):
        print("The sentence is: ", sentence.text)
        print("The entities are: ")
        for ent in pack.get("ft.onto.base_ontology.EntityMention", sentence):
            print(ent.text, ent.ner_type)
    print("Customized NLTKPOSTagger results: ")
    # print NLTK tagging results following token texts
    for token in pack.get(Token):
        print(f" {token.text}({token.pos})", end = " ")
    print()
```


## Learn More
The data-centric abstraction of Forte opens the gate to many other opportunities.
Not only does Forte allow engineers to develop reusable components easily, it further provides a simple way to develop composable ML modules. For example, Forte allows one to develop off-the-shelf processors from [3rd party toolkits](https://github.com/asyml/forte-wrappers) easily, build plug-and-play [data augmentation tools](https://asyml-forte.readthedocs.io/en/latest/code/data_aug.html), and allow one to build build reusable models as depicted below: 

![image](https://user-images.githubusercontent.com/1015991/164107427-66a5c9bd-a3ae-4d75-bfe2-24246e574e07.png)

To learn more about these, you can visit:
* [Examples](https://github.com/asyml/forte/tree/master/examples)
* [Documentation](https://asyml-forte.readthedocs.io/)
* Currently we are working on some interesting [tutorials](https://asyml-forte.readthedocs.io/en/latest/index_toc.html), stay tuned for a full set of documentation on how to do NLP with Forte!


## Contributing
Forte was originally developed in CMU and is actively contributed by [Petuum](https://petuum.com/) in collaboration with other institutes. This project is part of the [CASL Open Source](http://casl-project.ai/) family.

If you are interested in making enhancement to Forte, please first go over our [Code of Conduct](https://github.com/asyml/forte/blob/master/CODE_OF_CONDUCT.md) and [Contribution Guideline](https://github.com/asyml/forte/blob/master/CONTRIBUTING.md)

## About

### Supported By

<p align="center">
   <img src="https://asyml.io/assets/institutions/cmu.png", width="200" align="top">
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src="https://www.ucsd.edu/_resources/img/logo_UCSD.png" width="200" align="top">
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   <img src="https://raw.githubusercontent.com/asyml/forte/master/docs/_static/img/Petuum.png" width="200" align="top">
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p>

### The CASL Eco-System
![image](https://user-images.githubusercontent.com/1015991/164106557-13dd3781-95d6-42db-b90d-1685679184fe.png)

### License

[Apache License 2.0](https://github.com/asyml/forte/blob/master/LICENSE)

