.. role:: hidden
    :class: hidden-section

Data Augmentation
******************************


Data Augmentation Processors
================================

:hidden:`ReplacementDataAugmentProcessor`
---------------------------------------------
.. autoclass:: forte.processors.data_augment.base_data_augment_processor.ReplacementDataAugmentProcessor

:hidden:`DataSelector`
---------------------------------------------
.. autoclass:: forte.processors.base.data_selector_for_da.BaseElasticSearchDataSelector
    :members:


:hidden:`UDAIterator`
---------------------------------------------
.. autoclass:: forte.processors.data_augment.algorithms.UDA.UDAIterator
    :members:


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
    
:hidden:`MachineTranslator`
---------------------------------
.. autoclass:: forte.processors.data_augment.algorithms.machine_translator.MachineTranslator
    :members:

:hidden:`MarianMachineTranslator`
-----------------------------------
.. autoclass:: forte.processors.data_augment.algorithms.machine_translator.MarianMachineTranslator
    :members:

:hidden:`BackTranslationOp`
------------------------------
.. autoclass:: forte.processors.data_augment.algorithms.back_translation_op.BackTranslationOp
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

.. autoclass:: forte.models.da_rl.TexarBertMetaModule
    :members:
