.. role:: hidden
    :class: hidden-section

Data Augmentation
******************************


Data Augmentation Processors
================================


:hidden:`BaseDataAugmentProcessor`
---------------------------------------------
.. autoclass:: forte.processors.data_augment.base_data_augment_processor.BaseDataAugmentProcessor

:hidden:`ReplacementDataAugmentProcessor`
---------------------------------------------
.. autoclass:: forte.processors.data_augment.base_data_augment_processor.ReplacementDataAugmentProcessor


:hidden:`BaseElasticSearchDataSelector`
---------------------------------------------
.. autoclass:: forte.processors.base.data_selector_for_da.BaseElasticSearchDataSelector
    :members:

:hidden:`RandomDataSelector`
---------------------------------------------
.. autoclass:: forte.processors.base.data_selector_for_da.RandomDataSelector
    :members:


:hidden:`QueryDataSelector`
---------------------------------------------
.. autoclass:: forte.processors.base.data_selector_for_da.QueryDataSelector
    :members:



:hidden:`UDAIterator`
---------------------------------------------
.. autoclass:: forte.processors.data_augment.algorithms.UDA.UDAIterator
    :members:



:hidden:`RandomSwapDataAugmentProcessor`
---------------------------------------------
.. autoclass:: forte.processors.data_augment.algorithms.eda_processors.RandomSwapDataAugmentProcessor
    :members:

:hidden:`RandomInsertionDataAugmentProcessor`
---------------------------------------------
.. autoclass:: forte.processors.data_augment.algorithms.eda_processors.RandomInsertionDataAugmentProcessor
    :members:

:hidden:`RandomDeletionDataAugmentProcessor`
---------------------------------------------
.. autoclass:: forte.processors.data_augment.algorithms.eda_processors.RandomDeletionDataAugmentProcessor
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
.. autoclass:: forte.processors.data_augment.algorithms.sampler.Sampler
    :members:


:hidden:`UniformSampler`
------------------------------------
.. autoclass:: forte.processors.data_augment.algorithms.sampler.UniformSampler
    :members:

:hidden:`UnigramSampler`
------------------------------------
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

:hidden:`TypoReplacementOp`
----------------------------
.. autoclass:: forte.processors.data_augment.algorithms.typo_replacement_op.TypoReplacementOp
    :members:

:hidden:`CharacterFlipOp`
----------------------------
.. autoclass:: forte.processors.data_augment.algorithms.character_flip_op.CharacterFlipOp
    :members:

:hidden:`WordSplittingOp`
----------------------------
.. autoclass:: forte.processors.data_augment.algorithms.word_splitting_processor.RandomWordSplitDataAugmentProcessor
    :members:

:hidden:`BaseDataAugmentationOp`
-----------------------------------
.. autoclass:: forte.processors.data_augment.algorithms.base_data_augmentation_op.BaseDataAugmentationOp
    :members:

:hidden:`EmbeddingSimilarityReplacementOp`
------------------------------------------
.. autoclass:: forte.processors.data_augment.algorithms.embedding_similarity_replacement_op.EmbeddingSimilarityReplacementOp
    :members:

:hidden:`UniformTypoGenerator`
--------------------------------
.. autoclass:: forte.processors.data_augment.algorithms.typo_replacement_op.UniformTypoGenerator
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
