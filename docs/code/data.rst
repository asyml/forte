.. role:: hidden
    :class: hidden-section

Data
*******

Ontology
==========

top
----------
.. autoclass:: forte.data.ontology.top.Span
    :members:

.. autoclass:: forte.data.ontology.top.Entry
    :members:

.. autoclass:: forte.data.ontology.top.Annotation
    :members:

.. autoclass:: forte.data.ontology.top.Link
    :members:

.. autoclass:: forte.data.ontology.top.Group
    :members:

base_ontology
---------------

.. autoclass:: forte.data.ontology.base_ontology.Document
    :members:

.. autoclass:: forte.data.ontology.base_ontology.Sentence
    :members:

.. autoclass:: forte.data.ontology.base_ontology.Token
    :members:

.. autoclass:: forte.data.ontology.base_ontology.EntityMention
    :members:

.. autoclass:: forte.data.ontology.base_ontology.PredicateMention
    :members:

.. autoclass:: forte.data.ontology.base_ontology.PredicateArgument
    :members:

.. autoclass:: forte.data.ontology.base_ontology.PredicateLink
    :members:

.. autoclass:: forte.data.ontology.base_ontology.CoreferenceMention
    :members:

.. autoclass:: forte.data.ontology.base_ontology.CoreferenceGroup
    :members:

conll03_ontology
-----------------

.. autoclass:: forte.data.ontology.conll03_ontology.Token
    :members:

ontonotes_ontology
------------------

.. autoclass:: forte.data.ontology.ontonotes_ontology.Token
    :members:

.. autoclass:: forte.data.ontology.ontonotes_ontology.Sentence
    :members:

.. autoclass:: forte.data.ontology.ontonotes_ontology.PredicateMention
    :members:

relation_ontology
------------------

.. autoclass:: forte.data.ontology.relation_ontology.RelationLink
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

:hidden:`BaseMeta`
------------------------
.. autoclass:: forte.data.base_pack.BaseMeta
    :members:

:hidden:`Meta`
------------------------
.. autoclass:: forte.data.data_pack.Meta
    :members:

:hidden:`InternalMeta`
------------------------
.. autoclass:: forte.data.base_pack.InternalMeta
    :members:

:hidden:`BaseIndex`
------------------------
.. autoclass:: forte.data.base_pack.BaseIndex
    :members:

:hidden:`DataIndex`
------------------------
.. autoclass:: forte.data.data_pack.DataIndex
    :members:

Readers
==========

:hidden:`BaseReader`
------------------------
.. autoclass:: forte.data.readers.base_reader.BaseReader
    :members:

:hidden:`PackReader`
------------------------
.. autoclass:: forte.data.readers.base_reader.PackReader
    :members:

:hidden:`StringReader`
------------------------
.. autoclass:: forte.data.readers.string_reader.StringReader
    :members:

:hidden:`MonoFileReader`
------------------------
.. autoclass:: forte.data.readers.file_reader.MonoFileReader
    :members:

:hidden:`PlainTextReader`
--------------------------
.. autoclass:: forte.data.readers.plaintext_reader.PlainTextReader
    :members:

:hidden:`CoNLL03Reader`
------------------------
.. autoclass:: forte.data.readers.conll03_reader.CoNLL03Reader
    :members:

:hidden:`OntonotesReader`
--------------------------
.. autoclass:: forte.data.readers.ontonotes_reader.OntonotesReader
    :members:

Batchers
==========

:hidden:`Batcher`
------------------------
.. autoclass:: forte.data.batchers.Batcher
    :members:

:hidden:`ProcessingBatcher`
------------------------------
.. autoclass:: forte.data.batchers.ProcessingBatcher
    :members:

:hidden:`TexarBatcher`
------------------------
.. autoclass:: forte.data.batchers.TexarBatcher
    :members:


io_utils
==========

:hidden:`batch_instances`
----------------------------------
.. autofunction:: forte.data.io_utils.batch_instances

:hidden:`merge_batches`
----------------------------------
.. autofunction:: forte.data.io_utils.merge_batches

:hidden:`slice_batch`
----------------------------------
.. autofunction:: forte.data.io_utils.slice_batch

