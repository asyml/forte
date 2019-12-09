# Ontology Configuration #

Welcome! Forte uses _Ontologies_ to allow interaction between different NLP concepts, which is, simply put, a description of concepts and how they relate to each other. Forte allows you to define your project ontology in JSON format. The JSON ontology is then converted to Python classes automatically using Forte Ontology Generation feature.

This _Ontology Configuration_ tutorial teaches how to:
* Define a simple ontology config for your NLP project using JSON.
* Define multiple ontology configs and the dependencies between them.
* Generate the corresponding Python ontologies automatically and use them in your project. 

Overview of Forte concepts used in this tutorial:
* *Ontology* - A set of entries, one ontology can span one or more python 
modules. The modules generally belong to the package `ft.onto`.
* *Entry* - An entry corresponds to one NLP unit in the document, for instance, an annotated sequence or relationship between annotated sequences. `Token`, `Sentence` and `DependencyLink` are some examples of entries. One entry defined in the config is used to generate one python class.
* *Attribute* - An attribute generally corresponds to a label or property associated with an entry, like, `pos_tag` for the entry `Token`.
* *Top Entry* - Top entries are a set of entries defined in Forte, in the module `forte.data.ontology.base.top`. All user-defined entries should extend one of the top entries.
 
### Simple Ontology config ###
A simple example user-defined ontology looks like the following:
```json
{
    "name": "simple_ontology",
    "description": "",
    "import_paths": [
        "upper_ontology_config.json"
    ],
    "definitions": [
        {
            "namespace": "ft.onto.simple_ontology.Token",
            "parent_type": "ft.onto.upper_ontology.Token",
            "description": "",
            "attributes": [
                {
                    "name": "related_tokens",
                    "description": "Tokens related to the current token",
                    "type": "List",
                    "item_type": "str"
                },
                {
                    "name": "string_features",
                    "description": "Miscellaneous string features",
                    "type": "List",
                    "item_type": "ft.onto.simple_ontology.Token"
                }
            ]
        }
    ]
}
```
#### Breakdown of the Simple Example Ontology ####
The skeleton of the json schema looks like the following:
```json
{
  "name": "simple_ontology",
  "description": "Simple Ontology",
  "prefix": "ft.onto",
  "import_paths": [
  
  ],
  "definitions": [
  
  ]
}
```
- The `name` and `description` are annotation keywords meant for descriptive
purposes only.
- The `prefix` is an optional keyword that refers to the package name for all
the entry names and types. The default value of `prefix` is `ft.onto`. 
It is to be specified only when the user prefers to use a custom package name
for the generated ontology.
- `import_paths` is an optional keyword. It is used to define a list of json
configs paths that the current config might depend on.
    - The entries of the current config can use the entries of the imported
    configs in type specifications or defining parent classes. 
    - The import paths either be absolute paths or relative to the directory of 
    the current config, current working directory, or to one of the
    user-provided ``config_paths`` (see [usage](#usage).)
- `definitions:` is a list of entry definition, where each entry is represented
as a dictionary. For each entry, one class will be generated. The keywords of an
entry are explained in the next section.

##### ```definitions``` #####
For each definition - 
* The `entry_name` keyword defines the name of the entry. It should be 
of the form ```<module_name>.<entry_name>```.
    * The package name is `ft.onto` by default (unless, the ``prefix`` keyword 
    is defined at the top level of the config). It is used to generate the package directory tree.
    * The `<module_name>` would be the generated file name in which the entry would be placed. Note that entries defined in the same
    config can have module names that are different from each other.
    * The `<entry_name>` would be used as the generated class name.
 * The `parent_type`: Defines the base class of the generated entry class. All the user-defined entries should inherit either any of the top entries or one of the other user-defined entries.
 * `description`: Optional keyword used to define description of the entry to be used as the comment of the generated Python class if provided.
 * `attributes`: List of attributes that would be used as instance variables of the generated class. Each keyword for an attribute is defined in the next section.

##### ```attributes``` #####
For each attribute - 
* The `name` keyword defines the name of the property unique to the entry.
* `description`: Optional keyword used to define description of the entry to be used as the comment of the generated Python class if provided.
* `type: str`: Type of the attribute. Currently supported types are:
    * Primitive types - `int`, `float`, `str`, `bool`
    * Composite types - `List`
    * Top entries - The attributes can be of the type base
    entries (defined in the `top` module) and can be directly referred by the
     class name.
    * User-defined types - The attributes can be of the type entries that are
    user defined, and are specified as their namespaces. These entries could be
    defined (a) in the same config (b) any of the imported configs.
* `item_type: str`: If the `type` of the property is one of the composite types, then `item_type` can defines the type of the items contained in the property. As of now, we only support arrays of uniform types.

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
  
