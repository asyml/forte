Pipeline
==========
From a data source to a processing results, we want a uniform module that mangaes the workflow. For instance, given a data source in `txt` file for machine translation tasks, we want to read it from the file and use model to generate the translated text. Let's consider the modules doing these tasks as :class:`PipelineComponent`. For now, we focus on :class:`Pipeline` which contains :class:`PipelineComponent` and how it run through the task.

Let's check out a psudocode for setting up and running a pipeline.

.. code-block:: python

    pipeline: Pipeline = Pipeline[DataPack]() # intialize a pipeline
    pipeline.set_reader(SomePipelineComponent())
    pipeline.add(SomePipelineComponent())
    pipeline.add(SomePipelineComponent())
    pipeline.run(data_source) # it will call `initialize()` internally to initialize all :class:`PipelineComponent` in the pipeline.


As we can see, after initialize a pipeline, we set :class:`PipelineComponent` as reader which is the begin of the workflow and add :class:`PipelineComponent` into the workflow and then call `run` on the data source. :class:`PipelineComponent` are sorted in the order of adding internally which is same as the workflow order. It's a modular way of managing/running workflow.


In the actual machine translation task, we have the follow code to initialize a pipeline.

.. code-block:: python

    pipeline: Pipeline = Pipeline[DataPack]() # intialize a pipeline
    pipeline.set_reader(PlainTextReader())
    pipeline.add(MachineTranslationProcessor())
    pipeline.run(data_source) # it will call `initialize()` internally

:class:`PipelineComponent` can be reader, processor and selector. We can take a deeper look in the next sections.
