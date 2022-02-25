Pipeline
==========
From a data source to an inference results, we want a uniform module that mangages the workflow. For instance, given a data source in ``txt`` file for machine translation tasks, we want to read it from the file and use model to generate the translated text. Let's consider the modules doing these tasks as :class:`PipelineComponent`. Then, we focus on :class:`Pipeline` which contains :class:`PipelineComponent` and how it run through the task.

Let's check out a pseudocode for setting up and running a pipeline.

.. code-block:: python

    pipeline: Pipeline = Pipeline[DataPack]() # intialize a pipeline
    pipeline.set_reader(SomePipelineComponent())
    pipeline.add(SomePipelineComponent())
    pipeline.add(SomePipelineComponent())
    pipeline.run(data_source) # it will call `initialize()` internally to initialize all :class:`PipelineComponent` in the pipeline.


As we can see, after initialize a pipeline, we set :class:`PipelineComponent` as reader which is the beginning of the workflow and add :class:`PipelineComponent` into the workflow and then call ``run()`` on the data source. :class:`PipelineComponent` keeps the order of adding internally, and it is same as the workflow order. As we can see the whole pipeline setup is easy and clean as it's a modular way of managing/running workflow.


In the actual <machine translation task `../notebook_tutorial/wrap_MT_inference_pipeline.ipynb`>_, we have the following code to initialize a pipeline.

.. code-block:: python

    pipeline: Pipeline = Pipeline[DataPack]() # intialize a pipeline
    pipeline.set_reader(PlainTextReader())
    pipeline.add(MachineTranslationProcessor())
    pipeline.run(data_source) # it will call `initialize()` internally

:class:`PipelineComponent` can be reader, processor and selector. We can take a deeper look in the next sections.



For a more advanced usage:
https://github.com/asyml/forte/blob/7dc6e6c7d62d9a4126bdfc5ca02d15be3ffd61ca/forte/common/resources.py#L27
