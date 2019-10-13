# Ontology Configuration #

Welcome! Forte uses _Ontologies_ to allow interaction between different NLP concepts, which is, simply put, a description of concepts and how they relate to each other. Forte allows you to define your project ontology in JSON format. The JSON ontology is then converted to Python classes automatically using Forte Ontology Generation feature.

This _Ontology Configuration_ tutorial teaches how to:
* Define a simple ontology config for your NLP project using JSON.
* Define multiple ontology configs and the dependencies between them.
* Generate the corresponding Python ontologies automatically and use them in your project. 

Overview of Forte concepts used in this tutorial:
* *Ontology* - A set of entries, one ontology can span one or multiple python modules. The modules always belong to the package `forte.data.ontology`.
* *Entry* - An entry corresponds to one NLP unit to be annotated, or defines relationship between other entries. `Token`, `Sentence` and `DependencyLink` are some examples of entries. generates one python class.
* *Attribute* - An attribute generally corresponds to an annotation to an entry, like, `pos_tag` for the entry `Token`.
* *Base Entry* - Base entries are a set of entries defined in Forte, in the module `forte.data.ontology.base.top`. All user-defined entries should extend one of the base entries.
 
### Simple Ontology config ###
A simple example user-defined ontology looks like the following:
```json
{
    "ontology_name": "example_ontology",
    "entry_definitions": [
        {
            "entry_name": "forte.data.ontology.example_ontology.Token",
            "parent_entry": "forte.data.ontology.example_import_ontology.Token",
            "entry_description": "Just a string of contiguous characters.",
            "attributes": [
                {
                    "attribute_name": "string_features",
                    "attribute_description": "Miscellaneous string features",
                    "attribute_type": "List",
                    "element_type": "str"
                },
                ...
            ]
        }
    ]
}
```
#### Breakdown of the Simple Example Ontology ####
A json ontology has the filename of the form ``<ontology_name>_ontology_config.json``. The expected type for each key and explanation is provided below. The ontology contains these top level keys:
* `ontology_name: str`: Could be any unique name that defines the current ontology.
* `entry_definitions: List[Dict]`: List of entries, where each entry is represented as a dictionary. For each entry, a class will be generated. Each key is explained in detail in the next section. 

##### ```entry_definitions``` #####
* `entry_name`: A unique full name of the entry class. It should be of the form ```<package_name>.<module_name>.<entry_name>``` or ```<module_name>.<entry_name>```.
    * If the `<package_name>` is not provided, it is assumed to be `forte.data.ontology`. It is used to generate the package directory tree.
    * The `<module_name>` would be the generated file name in which the entry would be placed.
    * The `<entry_name>` would be used as the generated class name.
 * `parent_entry: str`: All the user-defined entries should inherit either any of the base entries or one of the other user-defined entries. The parent class would be used to initialize the arguments of the *\_\_init__* function.
 * `entry_description: Optional[str]`: String description of the entry to be used as the docstring of the generated Python class if provided.
 * `attributes: Optional[List[Dict]]`: List of attributes that would be used as instance variables of the generated class. Each attribute is defined in the next section.

##### ```attributes``` #####
* `attribute_name: str`: Name of the attribute unique to the entry.
* `attribute_description: Optional[str]`: String description of the attribute to be used as the docstring of the generated Python class if provided.
* `attribute_type: str`: Type of the attribute. Currently supported types are:
    * Primary types - `int`, `float`, `str` 
    * Composite types - `List`
* `element_type: str`: If the `attribute_type` is one of the composite types, the it defines the type of the elements in the attribute. Currently, `int`, `float`, `str` are supported.
### Multiple Ontology config ###
<!-- TODO: write about the `imports` and how different json files interact with each other -->
### Steps to generate the ontology ###
* Write the json config(s) as per the instructions in the previous sections. Let the base json config be defined in the path ``demo_ontology_config.json``.
* Initialize the Ontology Code Generator
    ```python
  import forte
  ontology_generator = forte.data.ontology.OntologyCodeGenerator()
    ```
* Pass the base json file in the ``generate_ontology`` function.
    ```python
  json_file_path = 'demo_ontology_config.json'
  destination_dir = 'destination_dir'
  
  destination_dir = ontology_generator.generate_ontology(json_file_path, destination_dir, dry_run=False)
    ```
  > The generated python config resides in the returned ``destination_dir``. The directory tree of the `destination_dir` looks like the following -
  

    .
    ├── ...
    ├── destination_dir
    │   ├── generated-files
    │   │   ├── forte                         # package name - forte.data.ontology
    │   │   │   ├── data
    │   │   │   │   ├── ontology
    │   │   │   │   ├── simple_ontology.py    # Contains entries belonging to the module, simple_ontology
    │   │   │   │   ├── upper_ontology.py     # Contains entries belonging to the module, upper_ontology
    └── ...
      
> If a file or directory named `generated-files` already exists in the `destination_dir`, an error would be thrown.

* The ``generate_ontology`` function also allows you to set the `dry_run=True`. In this case, the input `destination_dir` is ignored, and the `destination_dir` returned is the path to a temporary directory.

* Add the ``generated-files`` in the PYTHONPATH and add directory as the sources root for the IDE to be able to identify the packages.
<!-- TODO: write the steps to clean the generated code -->
  
