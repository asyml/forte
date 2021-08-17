.. role:: hidden
    :class: hidden-section

Training System
******************************

Forte promotes the convention to separate data pre-processing (Domain Dependent)
and actual training process. This is simply done by creating an intermediate
layer to extract raw features from data packs. In this documentation, we will
visit several components in this system, which includes:

* `Train Preprocessor` that defines the structure of this process.
* `Extractor` that extracts from data to features back and forth.
* `Converter` that creates matrices.
* `Predictor` that builds data pack from model output automatically.
* `Evaluator` that conducts evaluation on the resulting pack.

Train Preprocessor
================================

.. autoclass:: forte.train_preprocessor.TrainPreprocessor
    :members:


Converter
================================

.. autoclass:: forte.data.converter.converter.Converter
    :members:


Extractor
================================

:hidden:`BaseExtractor`
---------------------------------------
.. autoclass:: forte.data.base_extractor.BaseExtractor
    :members:

:hidden:`AttributeExtractor`
---------------------------------------
.. autoclass:: forte.data.extractors.attribute_extractor.AttributeExtractor
    :members:

:hidden:`CharExtractor`
---------------------------------------
.. autoclass:: forte.data.extractors.char_extractor.CharExtractor
    :members:

:hidden:`BioSeqTaggingExtractor`
---------------------------------------
.. autoclass:: forte.data.extractors.seqtagging_extractor.BioSeqTaggingExtractor
    :members:

Predictor
================================
.. autoclass:: forte.processors.base.batch_processor.Predictor
    :members:

Feature
================================

.. autoclass:: forte.data.converter.Feature
    :members:

Evaluation
**********

Base Evaluator
==============

.. autoclass:: forte.evaluation.base.base_evaluator.Evaluator
    :members:

Task Evaluators
===============

.. autoclass:: forte.evaluation.ner_evaluator.CoNLLNEREvaluator
    :members:
