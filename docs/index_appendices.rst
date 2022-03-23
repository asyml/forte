.. role:: hidden
    :class: hidden-section


Core Design Principles
==========================


The core design principle of Forte is the abstraction of NLP concepts and machine learning models,
which provides better separation between data, model and tasks, but enables interactions
between different components of the pipeline. Based on this, we make Forte:

* **Composable**: Forte helps users to decompose a problem into *data*, *models* and *tasks*. The tasks can further be divided into sub-tasks. A complex use case can be solved by composing heterogeneous modules via straightforward python APIs or declarative configuration files. The components (e.g. models or tasks) in the pipeline can be flexibly swapped in and out, as long as the API contracts are matched. The approach greatly improves module reusability, enables fast development and makes the library flexible for user needs.

* **Generalizable and Extensible**: Forte promotes generalization to support not only a wide range of NLP tasks, but also extensible for new tasks or new domains. In particular, Forte provides the *Ontology* system that helps users define types according to their tasks. Users can simply specify the type declaratively through JSON files. Our Code Generation tool will automatically generate python files ready to be used into your project. Check out our `Ontology Generation documentation <toc/ontology_generation.html>`_ for more details.

* **Transparent Data Flow**: Central to Forte's composable architecture is a universal data format that supports seamless data flow between different steps. Forte advocates a transparent data flow to facilitate flexible process intervention and simple pipeline control. Combined with the general data format, Forte makes a perfect tool for data inspection, component swapping and result sharing. This is particularly helpful during team collaborations!

.. image:: _static/img/forte_arch.png

.. image:: _static/img/forte_results.png


Glossary
============
.. toctree::
   :maxdepth: 2

* DataPack: a data class that stores structured data and supports efficient data retrieval.
    -  `DataPack Example <https://github.com/asyml/forte/blob/master/docs/tutorial/handling_structued_data.ipynb>`_
    - API: :class:`~forte.data.data_pack.DataPack`

* Pipeline: an inference system that contains a set of processing components.
    - `Pipeline Example <https://github.com/asyml/forte/tree/master/examples/pipelines>`_
    - API: :class:`~forte.pipeline.Pipeline`

* Ontology: a system that defines the relations between NLP annotations, for example, the relation between words and documents, or between two words.
    - `An ontology Example <https://github.com/asyml/forte/tree/master/examples/ontology>`_
    - `An ontology tutorial <https://github.com/asyml/forte/blob/0c1dec1311f27eae150287a8aa405632b265e03e/docs/tutorial/ontology_generation.md>`_



.. rst-class:: page-break


Examples
==========
.. toctree::
    :maxdepth: 2

Rich examples are included to demonstrate the use of Forte, including
 implementation of cutting-edge models/algorithms and system construction.

More examples are continuously added...


* `Data Reading: Showcasing how to read structured data. <https://github.com/asyml/forte/tree/master/examples/wiki_parser>`_
* `Serialization: Showcasing how to serialize and deserialize data. <https://github.com/asyml/forte/tree/master/examples/serialization>`_
* `NER: Train a LSTM-CRF named entity recognizer. <https://github.com/asyml/forte/tree/master/examples/ner>`_
* `BERT Passage Reranker <https://github.com/asyml/forte/tree/master/examples/passage_reranker>`_
* `Chat Bot: This example showcases the use of Forte to build a retrieval-based chatbot and perform text analysis on the retrieved results.  <https://github.com/asyml/forte/tree/master/examples/chatbot>`_
* `Audio Reading: a simple speech processing example here to showcase forte's capability to support a wide range of audio processing tasks. <https://github.com/asyml/forte/tree/master/examples/audio>`_
* `Classification: a text classification example that support various format of table-like dataset <https://github.com/asyml/forte/tree/master/examples/classification>`_
* `Clinical Pipeline: a project handling clinical datasets shows how to make Forte and Stave work side by side. <https://github.com/asyml/forte/tree/master/examples/clinical_pipeline>`_
* `Content Rewriter: a example which rewrites the sentence based on the table given a table and a sentence. <https://github.com/asyml/forte/tree/master/examples/content_rewriter>`_
* `Data Augmentation: this example demonstrates the usage of forte/models/da_rl/MetaAugmentationWrapper, that wraps a BERT Masked Language Model data augmentation model to perform this RL adaptive learning with a BERT-based text classifier downstream model. <https://github.com/asyml/forte/tree/master/examples/data_augmentation>`_
* `SRL: a semantic role labeling example <https://github.com/asyml/forte/tree/master/examples/srl>`_
* `Tagging: an implementation of CNN-BiLSTM-CRF model, built on top of Texar and Pytorch <https://github.com/asyml/forte/tree/master/examples/tagging>`_
* `Twitter sentiment analysis: this example show the use of Forte to perform sentiment analysis on the user's retrieved tweets <https://github.com/asyml/forte/tree/master/examples/twitter_sentiment_analysis>`_
* `Visualization: visualize datapack data <https://github.com/asyml/forte/tree/master/examples/visualize>`_
