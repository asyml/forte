.. role:: hidden
    :class: hidden-section

Processors
***********


Base Processors
===================

:hidden:`BaseProcessor`
----------------------------
.. autoclass:: forte.processors.base.base_processor.BaseProcessor
    :members:

:hidden:`BaseBatchProcessor`
----------------------------
.. autoclass:: forte.processors.base.batch_processor.BaseBatchProcessor
    :members:

:hidden:`BatchProcessor`
----------------------------
.. autoclass:: forte.processors.base.batch_processor.BatchProcessor
    :members:

:hidden:`BasePackProcessor`
----------------------------
.. autoclass:: forte.processors.base,pack_processor.BasePackProcessor
    :members:
    :private-members: _process

:hidden:`PackProcessor`
----------------------------
.. autoclass:: forte.processors.base.pack_processor.PackProcessor
    :members:

Task Processors
===================

:hidden:`NLTKSentenceSegmenter`
----------------------------------
.. autoclass:: forte.processors.sentence_predictor.NLTKSentenceSegmenter
    :members:

:hidden:`NLTKWordTokenizer`
----------------------------------
.. autoclass:: forte.processors.tokenization_predictor.NLTKWordTokenizer
    :members:

:hidden:`NLTKPOSTagger`
----------------------------------
.. autoclass:: forte.processors.postag_predictor.NLTKPOSTagger
    :members:

:hidden:`CoNLLNERPredictor`
----------------------------------
.. autoclass:: forte.processors.ner_predictor.CoNLLNERPredictor
    :members:

:hidden:`SRLPredictor`
----------------------------------
.. autoclass:: forte.processors.srl_predictor.SRLPredictor
    :members:

:hidden:`Alphabet`
----------------------------------
.. autoclass:: forte.processors.vocabulary_processor.Alphabet
    :members:

:hidden:`VocabularyProcessor`
----------------------------------
.. autoclass:: forte.processors.vocabulary_processor.VocabularyProcessor
    :members:

:hidden:`CoNLL03VocabularyProcessor`
--------------------------------------
.. autoclass:: forte.processors.vocabulary_processor.CoNLL03VocabularyProcessor
    :members: