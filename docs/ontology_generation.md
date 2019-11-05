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
    "name": "simple_ontology",
    "description": "",
    "imports": {
        "upper_ontology": {
            "type": "ft.onto.upper_ontology"
        }
    },
    "definitions": {
        "Token" : {
            "namespace": "ft.onto.simple_ontology.Token",
            "type": {
                "$ref": "#/imports/upper_ontology/definitions/Token"
            },
            "description": "",
            "properties": {
                "related_tokens": {
                    "description": "Tokens related to the current token",
                    "type": "List",
                    "items": {
                        "type": "str"
                    }
                },
                "string_features": {
                    "description": "Miscellaneous string features",
                    "type": "List",
                    "items": {
                        "$ref": "#/definitions/Token"
                    }
                }
            }
        }
    }
}
```
#### Breakdown of the Simple Example Ontology ####
The JSON config loosely follows the vocabulary of [JSON Schema](json-schema.org).
##### Skeleton of the schema #####
```json
{
  "name": "simple_ontology",
  "description": "Simple Ontology",
  "imports": [
  ...
  ],
  "definitions": [
  ...
  ]
}
```
- The `name` and `description` are annotation keywords meant for descriptive
purposes only. They do not add any value to the ontology generation.
- `imports` are an optional keyword. They are used to define a list of json configs 
that the current config might depend on. This is an advanced concept, and can be 
skipped. Expanded in more detail in the [Specifying Dependencies](#specifying-dependencies) section.
- `definitions:` contain entry definition objects, where each entry definition is represented as a dictionary. For each entry, one class will be generated. The keywords of `definitions` are explained in the next section.

##### ```definitions``` #####
* The top level keyword defines the reference name of the entry class, which
can be used to refer to this entry from other configs or other parts of the same config.
* The `namespace: str` keyword contain the full namespace of the entry. It should be of the form ```<package_name>.<module_name>.<entry_name>``` or ```<module_name>.<entry_name>```.
    * The `<package_name>` is used to generate the package directory tree. If it is not provided, it is assumed to be `ft.onto`..
    * The `<module_name>` would be the generated file name in which the entry would be placed.
    * The `<entry_name>` would be used as the generated class name.
 * The `type: str`: Defines the base class of the entry class. All the user-defined entries should inherit either any of the base entries (defined in top.py) or one of the other user-defined entries. The parent class would be used to initialize the arguments of the *\_\_init__* function.
 * `description: Optional[str]`: String description of the entry to be used as the docstring of the generated Python class if provided.
 * `properties: Optional[Dict]`: List of attributes that would be used as instance variables of the generated class. Each keyword for the properties is defined in the next section.

##### ```properties``` #####
* The top level keyword defines the name of the property unique to the entry.
* `description: Optional[str]`: String description of the attribute to be used as the docstring of the generated Python class if provided.
* `type: Union[str, Dict]`: Type of the attribute. Currently supported types are:
    * Primary types - `int`, `float`, `str` 
    * Composite types - `List`
    * Referred types - An entry can be referred using referred types. The entries could be in the same config or any of the imported configs. More on this in the [Specifying References](specifying-references) section.
* `items: Dict`: If the `type` of the property is one of the composite types, then `items/type` can defines the type of the items contained in the property. As of now, we only support arrays of uniform types.

### Specifying References ###
The `type` keyword can contain a reference type of the format, `{"$ref": "#/definitions/Token"}`. The idea is to use the schema that is stored under the result of evaluating the pointer `/definitions/Token` where Token is the reference to the entry under the same JSON config.

We could also be looking in the imported configs for references. The format for doing so is `{"$ref": "#/imports/<import_reference>/definitions/<entry_reference>"}`. The `import_reference` are the references to the import config
defined in `imports` and `entry_reference` is the reference to the entry under the imported JSON config. The import chain can be
arbitrarily long. 

### Specifying Dependencies ###
- `imports` are defined at the top level of the config. They are used to define 
the path of the config that the current config might depend on.
    The format of `imports` object is as follows:
    ```json
    {
      "simple_ontology": {
        "type": "ft/onto/upper_ontology_config.json"
      }
    }
  ```
    Here, the top level keyword, `simple_ontology`, defines the 
    reference to the imported config file defined using the `type` keyword.

### Usage ###
* Write the json config(s) as per the instructions in the previous sections. Let the base json config be defined in the path ``demo_ontology_config.json``.

* Use the command `generate_ontology` (added during installation) to generate 
the ontology. 

 #### Generating the ontology ####
 Use ``create`` mode to generate ontology given a root JSON config. It prints the
 path of the folder in which the package is generated top level directory is 
 outputted. For example:
 ```bash
python generate_ontology create \
    --config forte/data/ontology/configs/example_ontology_config.json \
    --no_dry_run
 ```
 All the arguments of `generate_ontology create` are explained as below: 
 ```bash
 usage: generate_ontology create [-h] --config CONFIG [--no_dry_run]
                          [--dest_path DEST_PATH] [--top_dir TOP_DIR]
                          [--config_paths [CONFIG_PATHS [CONFIG_PATHS ...]]]

optional arguments:
  -h, --help            Show this help message and exit.
  
  --config CONFIG       Path to the oot JSON config.
  
  --no_dry_run          Generates the package tree in the `--dest_path` if 
                        specified. By default, the package tree is generated in 
                        the temporary directory and `--dest_path` is ignored.
                        
  --dest_path DEST_PATH
                        Destination directory provided by the user. Only used
                        when `--no_dry_run is specified`. The default destination
                        directory is the current working directory.
                        
  --top_dir TOP_DIR     Top level directory to be created inside the destination
                        directory. By default, it is set as `generated-files`.
                        
  --config_paths [CONFIG_PATHS [CONFIG_PATHS ...]]
                        Paths in which the root and imported config files are
                        to be searched.
 ```
 #### Cleaning the ontology ####
 Use `clean` mode of `generate_ontology` to clean the generated files from the 
 given directory. For example:
 ```bash
python generate_ontology.py clean --dir ./generated-files
``` 
  All the arguments of `generate_ontology clean` are explained as below:
 ```bash
 usage: generate_ontology clean [-h] [--force] --dir DIR

optional arguments:
  -h, --help  show this help message and exit
  
  --dir DIR   Generated files to be cleaned from this directory path.
  
  --force     If true, skips the interactive deleting offolders. Use with
              caution.
 ```
  > The generated python config resides in the returned ``destination_dir``. The directory tree of the `destination_dir` looks like the following -

    .
    ├── ...
    ├── destination_dir
    │   ├── generated-files
    │   │   ├── ft                            # package name - ft.onto
    │   │   │   ├── onto
    │   │   │   │   ├── simple_ontology.py    # Contains entries belonging to the module, 
    │   │   │   │   │                           simple_ontology
    │   │   │   │   ├── upper_ontology.py     # Contains entries belonging to the module, 
    │   │   │   │   │                           upper_ontology
    └── ...
     
> If a file or directory named `generated-files` already exists in the `destination_dir`, an error would be thrown.

* The ``generate_ontology`` function also allows you to set the `dry_run=True`. In this case, the input `destination_dir` is ignored, and the `destination_dir` returned is the path to a temporary directory.

* Add the ``generated-files`` in the PYTHONPATH and add directory as the sources root for the IDE to be able to identify the packages.
<!-- TODO: write the steps to clean the generated code -->
  
