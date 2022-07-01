.. role:: hidden
    :class: hidden-section

Data
*******

Ontology
==========

base
----------
.. autoclass:: forte.data.span.Span
    :members:

core
----------
:hidden:`Entry`
------------------------------
.. autoclass:: forte.data.ontology.core.Entry
    :members:

.. autoclass:: forte.data.ontology.core.BaseLink
    :members:

.. autoclass:: forte.data.ontology.core.BaseGroup
    :members:


top
----------
.. autoclass:: forte.data.ontology.top.Generics
    :members:

.. autoclass:: forte.data.ontology.top.Annotation
    :members:

.. autoclass:: forte.data.ontology.top.AudioAnnotation
    :members:

.. autoclass:: forte.data.ontology.top.Link
    :members:

.. autoclass:: forte.data.ontology.top.Group
    :members:

.. autoclass:: forte.data.ontology.top.MultiPackGeneric
    :members:

.. autoclass:: forte.data.ontology.top.MultiPackGroup
    :members:

.. autoclass:: forte.data.ontology.top.MultiPackLink
    :members:

.. autoclass:: forte.data.ontology.top.Query
    :members:

Packs
==========

:hidden:`BasePack`
------------------------
.. autoclass:: forte.data.base_pack.BasePack
    :members:

:hidden:`DataPack`
------------------------
.. autoclass:: forte.data.data_pack.DataPack
    :members:

:hidden:`MultiPack`
------------------------
.. autoclass:: forte.data.multi_pack.MultiPack
    :members:

:hidden:`BaseMeta`
------------------------
.. autoclass:: forte.data.base_pack.BaseMeta
    :members:




:hidden:`Meta`
------------------------
.. autoclass:: forte.data.data_pack.Meta
    :members:


:hidden:`DataIndex`
------------------------
.. autoclass:: forte.data.data_pack.DataIndex
    :members:



MultiPack
==========

:hidden:`MultiPackMeta`
------------------------
.. autoclass:: forte.data.multi_pack.MultiPackMeta
    :members:


:hidden:`MultiPack`
------------------------
.. autoclass:: forte.data.multi_pack.MultiPack
    :members:
    :noindex:

:hidden:`MultiPackLink`
------------------------
.. autoclass:: forte.data.multi_pack.MultiPackLink
    :members:
    :noindex:

:hidden:`MultiPackGroup`
------------------------
.. autoclass:: forte.data.multi_pack.MultiPackGroup
    :members:
    :noindex:




Readers
==========

:hidden:`BaseReader`
------------------------
.. autoclass:: forte.data.base_reader.BaseReader
    :members:

:hidden:`PackReader`
------------------------
.. autoclass:: forte.data.base_reader.PackReader
    :members:

:hidden:`MultiPackReader`
--------------------------
.. autoclass:: forte.data.base_reader.MultiPackReader
    :members:

:hidden:`CoNLL03Reader`
------------------------
.. autoclass:: forte.data.readers.conll03_reader.CoNLL03Reader
    :members:

:hidden:`ConllUDReader`
------------------------
.. autoclass:: forte.data.readers.conllu_ud_reader.ConllUDReader
    :members:

:hidden:`BaseDeserializeReader`
--------------------------------
.. autoclass:: forte.data.readers.deserialize_reader.BaseDeserializeReader
    :members:

:hidden:`RawDataDeserializeReader`
-----------------------------------
.. autoclass:: forte.data.readers.deserialize_reader.RawDataDeserializeReader
    :members:

:hidden:`RecursiveDirectoryDeserializeReader`
----------------------------------------------
.. autoclass:: forte.data.readers.deserialize_reader.RecursiveDirectoryDeserializeReader
    :members:

:hidden:`HTMLReader`
----------------------------
.. autoclass:: forte.data.readers.html_reader.HTMLReader
    :members:

:hidden:`MSMarcoPassageReader`
-------------------------------
.. autoclass:: forte.data.readers.ms_marco_passage_reader.MSMarcoPassageReader
    :members:

:hidden:`MultiPackSentenceReader`
----------------------------------
.. autoclass:: forte.data.readers.multipack_sentence_reader.MultiPackSentenceReader
    :members:

:hidden:`MultiPackTerminalReader`
----------------------------------
.. autoclass:: forte.data.readers.multipack_terminal_reader.MultiPackTerminalReader
    :members:

:hidden:`OntonotesReader`
--------------------------
.. autoclass:: forte.data.readers.ontonotes_reader.OntonotesReader
    :members:

:hidden:`PlainTextReader`
--------------------------
.. autoclass:: forte.data.readers.plaintext_reader.PlainTextReader
    :members:

:hidden:`ProdigyReader`
--------------------------
.. autoclass:: forte.data.readers.prodigy_reader.ProdigyReader
    :members:

:hidden:`RACEMultiChoiceQAReader`
----------------------------------
.. autoclass:: forte.data.readers.race_multi_choice_qa_reader.RACEMultiChoiceQAReader
    :members:

:hidden:`StringReader`
------------------------
.. autoclass:: forte.data.readers.string_reader.StringReader
    :members:

:hidden:`SemEvalTask8Reader`
------------------------------
.. autoclass:: forte.data.readers.sem_eval_task8_reader.SemEvalTask8Reader
    :members:

:hidden:`OpenIEReader`
------------------------
.. autoclass:: forte.data.readers.openie_reader.OpenIEReader
    :members:

:hidden:`SquadReader`
------------------------
.. autoclass:: forte.datasets.mrc.squad_reader.SquadReader
    :members:

:hidden:`ClassificationDatasetReader`
--------------------------------------
.. autoclass:: forte.data.readers.classification_reader.ClassificationDatasetReader
    :members:

Selector
=================
:hidden:`Selector`
--------------------------------------
.. autoclass:: forte.data.selector.Selector
    :members:

:hidden:`DummySelector`
--------------------------------------
.. autoclass:: forte.data.selector.DummySelector
    :members:

:hidden:`SinglePackSelector`
--------------------------------------
.. autoclass:: forte.data.selector.SinglePackSelector
    :members:

:hidden:`NameMatchSelector`
--------------------------------------
.. autoclass:: forte.data.selector.NameMatchSelector
    :members:

:hidden:`RegexNameMatchSelector`
--------------------------------------
.. autoclass:: forte.data.selector.RegexNameMatchSelector
    :members:

:hidden:`FirstPackSelector`
--------------------------------------
.. autoclass:: forte.data.selector.FirstPackSelector
    :members:

:hidden:`AllPackSelector`
--------------------------------------
.. autoclass:: forte.data.selector.AllPackSelector
    :members:







Index
=================
:hidden:`BaseIndex`
------------------------
.. autoclass:: forte.data.index.BaseIndex
    :members:




Store
=================
:hidden:`BaseStore`
--------------------------------------
.. autoclass:: forte.data.base_store.BaseStore
    :members:


Data Store
=================
:hidden:`DataStore`
--------------------------------------
.. autoclass:: forte.data.data_store.DataStore
    :members:



DataPack Dataset
=================

:hidden:`DataPackIterator`
------------------------------
.. autoclass:: forte.data.data_pack_dataset.DataPackIterator
    :members:

:hidden:`DataPackDataset`
------------------------------
.. autoclass:: forte.data.data_pack_dataset.DataPackDataset
    :members:

:hidden:`RawExample`
------------------------------
.. autoclass:: forte.data.data_pack_dataset.RawExample
    :members:

:hidden:`FeatureCollection`
------------------------------
.. autoclass:: forte.data.data_pack_dataset.FeatureCollection
    :members:




Batchers
==========

:hidden:`ProcessingBatcher`
------------------------------
.. autoclass:: forte.data.batchers.ProcessingBatcher
    :members:

:hidden:`FixedSizeDataPackBatcherWithExtractor`
---------------------------------------------------
.. autoclass:: forte.data.batchers.FixedSizeDataPackBatcherWithExtractor
    :members:

:hidden:`FixedSizeRequestDataPackBatcher`
---------------------------------------------------
.. autoclass:: forte.data.batchers.FixedSizeRequestDataPackBatcher
    :members:

:hidden:`FixedSizeMultiPackProcessingBatcher`
---------------------------------------------------
.. autoclass:: forte.data.batchers.FixedSizeMultiPackProcessingBatcher
    :members:

:hidden:`FixedSizeDataPackBatcher`
------------------------------------
.. autoclass:: forte.data.batchers.FixedSizeDataPackBatcher
    :members:




Caster
==========

:hidden:`Caster`
------------------------------
.. autoclass:: forte.data.caster.Caster
    :members:

:hidden:`MultiPackBoxer`
------------------------------
.. autoclass:: forte.data.caster.MultiPackBoxer
    :members:

:hidden:`MultiPackUnboxer`
------------------------------
.. autoclass:: forte.data.caster.MultiPackUnboxer
    :members:


Container
==========

:hidden:`EntryContainer`
------------------------------
.. autoclass:: forte.data.container.EntryContainer
    :members:


Types
===============
:hidden:`ReplaceOperationsType`
--------------------------------
.. autoclass:: forte.data.types.ReplaceOperationsType
    :members:

:hidden:`DataRequest`
------------------------------
.. autoclass:: forte.data.types.DataRequest
    :members:

:hidden:`MatrixLike`
------------------------------
.. autoclass:: forte.data.types.MatrixLike
    :members:



Data Utilities
===============

:hidden:`maybe_download`
----------------------------------
.. autofunction:: forte.data.data_utils.maybe_download

:hidden:`batch_instances`
----------------------------------
.. autofunction:: forte.data.data_utils_io.batch_instances

:hidden:`merge_batches`
----------------------------------
.. autofunction:: forte.data.data_utils_io.merge_batches

:hidden:`slice_batch`
----------------------------------
.. autofunction:: forte.data.data_utils_io.slice_batch

:hidden:`dataset_path_iterator`
----------------------------------
.. autofunction:: forte.data.data_utils_io.dataset_path_iterator


Entry Utilities
===============

:hidden:`create_utterance`
----------------------------------
.. autofunction:: forte.data.common_entry_utils.create_utterance

:hidden:`get_last_utterance`
----------------------------------
.. autofunction:: forte.data.common_entry_utils.get_last_utterance
