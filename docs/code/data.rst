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
.. autoclass:: forte.data.ontology.core.Entry
    :members:

.. autoclass:: forte.data.ontology.core.BaseLink
    :members:

.. autoclass:: forte.data.ontology.core.BaseGroup
    :members:


top
----------
.. autoclass:: forte.data.ontology.top.Annotation
    :members:

.. autoclass:: forte.data.ontology.top.Link
    :members:

.. autoclass:: forte.data.ontology.top.Group
    :members:

.. autoclass:: forte.data.ontology.top.MultiPackGroup
    :members:

.. autoclass:: forte.data.ontology.top.MultiPackLink
    :members:

.. autoclass:: forte.data.ontology.top.SubEntry
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

:hidden:`ProcessingBatcher`
------------------------------
.. autoclass:: forte.data.batchers.ProcessingBatcher
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

