Processor
==========

A pipeline component that wraps inference model and set up inference related work.

Related Readings
------------------

#. `Processor API <../code/processors.html>`_

Functions
----------

* ``initialize()``: Pipeline will call it at the start of processing. The processor will be initialized with ``configs``, and register global resources into :class:`forte.common.Resources`. The implementation should set up the states of the component.

    - ``default_configs`` is a class method that returns default configuration
      in a dictionary format. Parent reader class configuration will be merged
      or overwritten by child class.

        - ``default_configs`` usage example

            - To use an existing processor, User should check configurations
              from method ``default_configs`` of the particular processor
              used to
              find out what configurations can be customized. For example,
              suppose after checking `processor API <../code/processors.html>`_
              we decide to use
              :class:`~forte.processors.base.base_processor.BaseProcessor`.
              Then we need to check the source of
              :meth:`forte.processors.base.base_processor.BaseProcessor.default_configs()`
              and found that ``"overwrite"`` is a boolean configuration and we
              can set it to ``False`` in our customized configuration when we
              don't want the default configuration. The default configuration
              will be overwritten when we initialize the processor with our
              customized configuration.

            - To implement a new processor, User should check the appropriate
              processor to inherit from. One consideration is whether User
              wants to process a ``DataPack`` or a ``DataPack`` batch for
              each processing iteration. If it's the
              ``DataPack``, then User should inherit from
              :class:`~forte.processors.base.pack_processor.PackProcessor`.
              If it's the ``DataPack`` batch, then User should inherit from
              :class:`~forte.processors.base.batch_processor.BaseBatchProcessor`
              For example, in the implementation of
              :class:`~forte.processors.misc.vocabulary_processor.VocabularyProcessor`, it inherits
              from :class:`~forte.processors.base.pack_processor.PackProcessor`
              because it builds vocabulary from ``DataPack``. Then User can
              consider adding a new configuration field in
              ``default_configs()`` based on the needs
              or overwrite the configuration field from its parent class.
              It's just a simple consideration to explain the process of
              choosing the right processor, there are many other processors
              with more features that User can inherit from. User can refer to
              `Processors API <../code/processors.rst>`_ for more
              information.

    - ``resource`` is for advanced developer. It's an shared object that stores data accessible by all ``PipelineComponent`` in the pipeline.


* ``_process()``: The main function of the processor. The implementation should process the ``input_pack``, and conduct operations such as adding entries into the pack, or produce some side-effect such as writing data into the disk.



We also have plenty of written processors available to use. If you don't find
one suitable in your case, you can refer to pipeline examples, API or tutorials
to customize a new processor.
