.. role:: hidden
    :class: hidden-section

Processors
***********


Base Processors
===================

:hidden:`BaseProcessor`
----------------------------
.. autoclass:: nlp.pipeline.processors.base_processor.BaseProcessor
    :members:

:hidden:`BaseBatchProcessor`
----------------------------
.. autoclass:: nlp.pipeline.processors.batch_processor.BaseBatchProcessor
    :members:

:hidden:`BatchProcessor`
----------------------------
.. autoclass:: nlp.pipeline.processors.batch_processor.BatchProcessor
    :members:

:hidden:`BasePackProcessor`
----------------------------
.. autoclass:: nlp.pipeline.processors.pack_processor.BasePackProcessor
    :members:
    :private-members: _process

:hidden:`PackProcessor`
----------------------------
.. autoclass:: nlp.pipeline.processors.pack_processor.PackProcessor
    :members:

Task Processors
===================

:hidden:`NLTKSentenceSegmenter`
----------------------------------
.. autoclass:: nlp.pipeline.processors.impl.sentence_predictor.NLTKSentenceSegmenter
    :members:

:hidden:`NLTKWordTokenizer`
----------------------------------
.. autoclass:: nlp.pipeline.processors.impl.tokenization_predictor.NLTKWordTokenizer
    :members:

:hidden:`NLTKPOSTagger`
----------------------------------
.. autoclass:: nlp.pipeline.processors.impl.postag_predictor.NLTKPOSTagger
    :members:

:hidden:`CoNLLNERPredictor`
----------------------------------
.. autoclass:: nlp.pipeline.processors.impl.ner_predictor.CoNLLNERPredictor
    :members:

:hidden:`SRLPredictor`
----------------------------------
.. autoclass:: nlp.pipeline.processors.impl.srl_predictor.SRLPredictor
    :members:

:hidden:`Alphabet`
----------------------------------
.. autoclass:: nlp.pipeline.processors.impl.vocabulary_processor.Alphabet
    :members:

:hidden:`VocabularyProcessor`
----------------------------------
.. autoclass:: nlp.pipeline.processors.impl.vocabulary_processor.VocabularyProcessor
    :members:

:hidden:`CoNLL03VocabularyProcessor`
--------------------------------------
.. autoclass:: nlp.pipeline.processors.impl.vocabulary_processor.CoNLL03VocabularyProcessor
    :members: