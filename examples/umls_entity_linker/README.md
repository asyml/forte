# Pipeline Demo

The example in this folder builds a processor using a third-party tool called scispaCy.
We will then use this processor to find all the linked concept ids for each entity in the text.


# Description

## Install the dependencies

- Since we'll be using scispaCy, please install scispaCy library using

```bash
pip install scispaCy
```

or install from `requirements.txt file`

```bash
pip install -r requirements.txt
```

## Writing the Processor

The file `umls_entity_linker_processor.py` shows an example of writing a processor.

The class  `ScispaCyUMLSEntityLinker` extends `PackProcssor` which is the bast class for a processor that consumes packs.
We initialize the configs (defined in `default_configs` method) in the `initialize` method.

Each processor must override the `_process` function which defines the behavior for the processor. The function consumes packs and modifies the pack according to it's expected behavior.

In this example, we read the packs, use the scispaCy pipeline to extract entity information from the model and save the useful information in the pack.
To add the information in the pack, we use the pack's `add_entry()` function.  


## Running the pipeline

`umls_entity_linking_example.py` builds the following pipeline

Reader -> ScispaCyUMLSEntityLinker

The configuration for ScispaCyUMLSEntityLinker are initialized in the file itself and the processor is 
initialized during the pipeline creation.

We read the files from a string and we use the StringReader Processor to do that.

To see the pipeline in action, run 

```bash
python pipeline_string_example.py
```
 
