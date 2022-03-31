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

:hidden:`BasePackProcessor`
----------------------------------
.. autoclass:: forte.processors.base.pack_processor.BasePackProcessor
    :members:


:hidden:`BaseBatchProcessor`
-----------------------------------
.. autoclass:: forte.processors.base.batch_processor.BaseBatchProcessor
    :members:

:hidden:`PackingBatchProcessor`
-----------------------------------
.. autoclass:: forte.processors.base.batch_processor.PackingBatchProcessor
    :members:
    :private-members: _process

:hidden:`MultiPackBatchProcessor`
-----------------------------------
.. autoclass:: forte.processors.base.batch_processor.MultiPackBatchProcessor
    :members:

:hidden:`RequestPackingProcessor`
-----------------------------------
.. autoclass:: forte.processors.base.batch_processor.RequestPackingProcessor
    :members:

:hidden:`FixedSizeBatchProcessor`
-----------------------------------
.. autoclass:: forte.processors.base.batch_processor.FixedSizeBatchProcessor
    :members:


:hidden:`Predictor`
----------------------------
.. autoclass:: forte.processors.base.batch_processor.Predictor
    :members:
    :noindex:

Pack Processors
===================

:hidden:`PackProcessor`
----------------------------------
.. autoclass:: forte.processors.base.pack_processor.PackProcessor
    :members:



:hidden:`MultiPackProcessor`
----------------------------------
.. autoclass:: forte.processors.base.pack_processor.MultiPackProcessor
    :members:


Task Processors
===================

:hidden:`ElizaProcessor`
----------------------------------
.. autoclass:: forte.processors.nlp.eliza_processor.ElizaProcessor
    :members:

:hidden:`SubwordTokenizer`
----------------------------------
.. autoclass:: forte.processors.nlp.subword_tokenizer.SubwordTokenizer
    :members:


:hidden:`CoNLLNERPredictor`
----------------------------------
.. autoclass:: forte.processors.nlp.ner_predictor.CoNLLNERPredictor
    :members:

:hidden:`SRLPredictor`
----------------------------------
.. autoclass:: forte.processors.nlp.srl_predictor.SRLPredictor
    :members:

:hidden:`VocabularyProcessor`
----------------------------------
.. autoclass:: forte.processors.misc.vocabulary_processor.VocabularyProcessor
    :members:

:hidden:`Alphabet`
----------------------------------
.. autoclass:: forte.processors.misc.vocabulary_processor.Alphabet
    :members:

:hidden:`PeriodSentenceSplitter`
----------------------------------
.. autoclass:: forte.processors.misc.simple_processors.PeriodSentenceSplitter
    :members:

:hidden:`WhiteSpaceTokenizer`
----------------------------------
.. autoclass:: forte.processors.misc.simple_processors.WhiteSpaceTokenizer
    :members:

:hidden:`RemoteProcessor`
----------------------------------
.. autoclass:: forte.processors.misc.remote_processor.RemoteProcessor
    :members:

:hidden:`LowerCaserProcessor`
----------------------------------
.. autoclass:: forte.processors.misc.lowercaser_processor.LowerCaserProcessor
    :members:

:hidden:`DeleteOverlapEntry`
----------------------------------
.. autoclass:: forte.processors.misc.delete_overlap_entries.DeleteOverlapEntry
    :members:

:hidden:`AttributeMasker`
----------------------------------
.. autoclass:: forte.processors.misc.attribute_masking_processor.AttributeMasker
    :members:


:hidden:`AnnotationRemover`
----------------------------------
.. autoclass:: forte.processors.misc.annotation_remover.AnnotationRemover
    :members:
