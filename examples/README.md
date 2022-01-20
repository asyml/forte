This is a tutorial how to use a forte pipeline with customized modular forte 
classes to run your tasks.

`forte.pipeline` is a class functions as its name indicates. User needs to add
various readers or processors to process the data. There are many advantages about
this.
* all components are highly customizable by editing configurations. Usually, there is `default_configs` for each component and user can create their own configurations and then pass the configuration while adding the components into pipeline. For example, we can add a pipeline reader to pipeline in this way.
```python
reader_config = {...}
pl.set_reader(SomeDatasetReader(), config=reader_config)
```
* processors/models can be from many well-developed external open-source packages such as Huggingface, Spacy and nltk.
* For various types of data source, we can focus on reader that highly-customizable ontologies and user can adapt configuration files for their specific tasks and `DatasetReader` they choose.
  *  For example, user can customize input data fields and label data fields out of all data fields.
  *  Data source is not only limited to text data as we have various readers to different types of data including audio and images. Some are currently under development.
* After all pipeline components are ready, users can initialize pipeline by `pipeline.initialize()`
* We iterate data by `pipeline.process_dataset(data_path)`.
  * It's similar to `DataIterator` but it yields `DataPack` that contains both processed input data and data ontologies. Such a data class design could be adapted to a more complex ML training tasks.
