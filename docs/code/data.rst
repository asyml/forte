.. role:: hidden
    :class: hidden-section

Data
*******

Ontology
==========

top
----------
.. autoclass:: nlp.pipeline.data.ontology.top.Span
    :members:

.. autoclass:: nlp.pipeline.data.ontology.top.Entry
    :members:

.. autoclass:: nlp.pipeline.data.ontology.top.Annotation
    :members:

.. autoclass:: nlp.pipeline.data.ontology.top.Link
    :members:

.. autoclass:: nlp.pipeline.data.ontology.top.Group
    :members:

base_ontology
---------------

.. autoclass:: nlp.pipeline.data.ontology.base_ontology.Document
    :members:

.. autoclass:: nlp.pipeline.data.ontology.base_ontology.Sentence
    :members:

.. autoclass:: nlp.pipeline.data.ontology.base_ontology.Token
    :members:

.. autoclass:: nlp.pipeline.data.ontology.base_ontology.EntityMention
    :members:

.. autoclass:: nlp.pipeline.data.ontology.base_ontology.PredicateMention
    :members:

.. autoclass:: nlp.pipeline.data.ontology.base_ontology.PredicateArgument
    :members:

.. autoclass:: nlp.pipeline.data.ontology.base_ontology.PredicateLink
    :members:

.. autoclass:: nlp.pipeline.data.ontology.base_ontology.CoreferenceMention
    :members:

.. autoclass:: nlp.pipeline.data.ontology.base_ontology.CoreferenceGroup
    :members:

conll03_ontology
-----------------

.. autoclass:: nlp.pipeline.data.ontology.conll03_ontology.Token
    :members:

ontonotes_ontology
------------------

.. autoclass:: nlp.pipeline.data.ontology.ontonotes_ontology.Token
    :members:

.. autoclass:: nlp.pipeline.data.ontology.ontonotes_ontology.Sentence
    :members:

.. autoclass:: nlp.pipeline.data.ontology.ontonotes_ontology.PredicateMention
    :members:

relation_ontology
------------------

.. autoclass:: nlp.pipeline.data.ontology.relation_ontology.RelationLink
    :members:

Packs
==========

:hidden:`BasePack`
------------------------
.. autoclass:: nlp.pipeline.data.base_pack.BasePack
    :members:

:hidden:`DataPack`
------------------------
.. autoclass:: nlp.pipeline.data.data_pack.DataPack
    :members:

:hidden:`BaseMeta`
------------------------
.. autoclass:: nlp.pipeline.data.base_pack.BaseMeta
    :members:

:hidden:`Meta`
------------------------
.. autoclass:: nlp.pipeline.data.data_pack.Meta
    :members:

:hidden:`InternalMeta`
------------------------
.. autoclass:: nlp.pipeline.data.base_pack.InternalMeta
    :members:

:hidden:`BaseIndex`
------------------------
.. autoclass:: nlp.pipeline.data.base_pack.BaseIndex
    :members:

:hidden:`DataIndex`
------------------------
.. autoclass:: nlp.pipeline.data.data_pack.DataIndex
    :members:

Readers
==========

:hidden:`BaseReader`
------------------------
.. autoclass:: nlp.pipeline.data.readers.base_reader.BaseReader
    :members:

:hidden:`DataPackReader`
------------------------
.. autoclass:: nlp.pipeline.data.readers.base_reader.DataPackReader
    :members:

:hidden:`StringReader`
------------------------
.. autoclass:: nlp.pipeline.data.readers.string_reader.StringReader
    :members:

:hidden:`MonoFileReader`
------------------------
.. autoclass:: nlp.pipeline.data.readers.file_reader.MonoFileReader
    :members:

:hidden:`PlainTextReader`
--------------------------
.. autoclass:: nlp.pipeline.data.readers.plaintext_reader.PlainTextReader
    :members:

:hidden:`CoNLL03Reader`
------------------------
.. autoclass:: nlp.pipeline.data.readers.conll03_reader.CoNLL03Reader
    :members:

:hidden:`OntonotesReader`
--------------------------
.. autoclass:: nlp.pipeline.data.readers.ontonotes_reader.OntonotesReader
    :members:

Batchers
==========

:hidden:`Batcher`
------------------------
.. autoclass:: nlp.pipeline.data.batchers.Batcher
    :members:

:hidden:`ProcessingBatcher`
------------------------------
.. autoclass:: nlp.pipeline.data.batchers.ProcessingBatcher
    :members:

:hidden:`TexarBatcher`
------------------------
.. autoclass:: nlp.pipeline.data.batchers.TexarBatcher
    :members:


io_utils
==========

:hidden:`batch_instances`
----------------------------------
.. autofunction:: nlp.pipeline.data.io_utils.batch_instances

:hidden:`merge_batches`
----------------------------------
.. autofunction:: nlp.pipeline.data.io_utils.merge_batches

:hidden:`slice_batch`
----------------------------------
.. autofunction:: nlp.pipeline.data.io_utils.slice_batch

