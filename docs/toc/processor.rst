Processor
==========

A pipeline component that wraps inference model and set up inference related work.



Functions
----------

* ``initialize()``: Pipeline will call it at the start of processing. The processor will be initialized with ``configs``, and register global resources into ``resource``. The implementation should set up the states of the component.

* ``_process()``: The main function of the processor. The implementation should process the ``input_pack``, and conduct operations such as adding entries into the pack, or produce some side-effect such as writing data into the disk.



We also have plenty of written processors available to use. If you don't find one suitable in your case, you can refer to pipeline examples, API or tutorials to customize a new processor.
