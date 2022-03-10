Processor
==========

A pipeline component that wraps inference model and set up inference related work.

Related Readings
------------------

#. `Processor API <../code/processors.html>`_

Functions
----------

* ``initialize()``: Pipeline will call it at the start of processing. The processor will be initialized with ``configs``, and register global resources into :class:`forte.common.Resources`. The implementation should set up the states of the component.
    - User should check configurations from method `default_configs` of the particular processor used to find out what configurations can be customized. For example, suppose after checking `processor API <../code/processors.html>`_ we decide to use :class:`~forte.processors.base.base_processor.BaseProcessor`. Then we need to check the source of :meth:`forte.processors.base.base_processor.BaseProcessor.default_configs()` and found that ``"overwrite"`` is a boolean configuration and we can set it to ``False`` in our customized configuration when we don't want the default configuration. The default configuration will be overwritten when we initialize the processor with our customized configuration.

* ``_process()``: The main function of the processor. The implementation should process the ``input_pack``, and conduct operations such as adding entries into the pack, or produce some side-effect such as writing data into the disk.



We also have plenty of written processors available to use. If you don't find one suitable in your case, you can refer to pipeline examples, API or tutorials to customize a new processor.
