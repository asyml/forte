.. role:: hidden
    :class: hidden-section

Data Augmentation
******************************


Data Augmentation Processors
================================

:hidden:`ReplacementDataAugmentProcessor`
---------------------------------------------
.. autoclass:: forte.processors.base.data_augment_processor.ReplacementDataAugmentProcessor
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
