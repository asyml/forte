.. role:: hidden
    :class: hidden-section

Data Augmentation
******************************


Data Augmentation Processors
================================



Data Augmentation Ops
========================

:hidden:`TextReplacementOp`
-----------------------------------
.. autoclass:: forte.processors.data_augment.algorithms.text_replacement_op.TextReplacementOp
    :members:

:hidden:`DistributionReplacementOp`
------------------------------------
.. autoclass:: forte.processors.data_augment.algorithms.DistributionReplacementOp
    :members:

:hidden:`Sampler`
------------------------------------
.. autoclass:: forte.processors.data_augment.algorithms.sampler.UniformSampler
    :members:

.. autoclass:: forte.processors.data_augment.algorithms.sampler.UnigramSampler
    :members:

:hidden:`DictionaryReplacementOp`
---------------------------------------
.. autoclass:: forte.processors.data_augment.algorithms.dictionary_replacement_op.DictionaryReplacementOp
    :members:

:hidden:`Dictionary`
----------------------------
.. autoclass:: forte.processors.data_augment.algorithms.dictionary.Dictionary
    :members:

:hidden:`WordnetDictionary`
----------------------------
.. autoclass:: forte.processors.data_augment.algorithms.dictionary.WordnetDictionary
    :members:

Data Augmentation Models
========================================

:hidden:`Reinforcement Learning`
----------------------------------
.. autoclass:: forte.models.da_rl.MetaAugmentationWrapper
    :members:

.. autoclass:: forte.models.da_rl.MetaModule
    :members:
