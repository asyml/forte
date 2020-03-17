# Ontology Configuration #

Welcome! Forte allows one to define relations between the annotations, 
 using _Ontologies_. This allows we have rich interaction between different NLP 
concepts.

Forte allows you to define the ontology of your project 
in JSON format. The JSON ontology is then converted to Python classes 
automatically using the Forte Ontology Generation feature.

### A simple ontology config ###
Let us consider a simple ontology for documents of a pet shop. Let's we want to 
know which text talks about the `Pet`s, and which text talks about `Customer`s.
```json
{
    "name": "pet_shop_ontology",
    "description": "An Ontology Used to manage the pet shop assets and pets",
    "definitions": [
        {
            "entry_name": "ft.onto.pet_shop.Pet",
            "parent_entry": "forte.data.ontology.top.Annotation",
            "description": "Pets in the shop.",
            "attributes": [
                {
                    "name": "pet_type",
                    "type": "str"
                }
            ]
        },
        {
            "entry_name": "ft.onto.pet_shop.Customer",
            "parent_entry": "forte.data.ontology.top.Annotation",
            "description": "Owner of pets.",
            "attributes": [
                {
                    "name": "name",
                    "type": "str"
                },
                {
                    "name": "pets",
                    "description": "List of pets the customer have.",
                    "type": "List",
                    "item_type": "ft.onto.pet_shop.Pet"
                }
            ]
        }
    ]
}
```

At Forte's root directory, try run the following command:
```shell
generate_ontology create --config ontology_definitions/example_ontology.json --dest_path .
```
You should be able to see the generated code for this Pet Shop ontology.





This _Ontology Configuration_ tutorial teaches how to:
* Define a simple ontology config for your NLP project using JSON.
* Define multiple ontology configs and the dependencies between them.
* Generate the corresponding Python classes automatically and use them in 
your project.

Overview of Forte concepts used in this tutorial:
* *Entry* - An entry corresponds to one NLP unit in the document, for instance, 
an annotated sequence or relationship between annotated sequences. `Token`, 
`Sentence` and `DependencyLink` are some examples of entries. One entry defined 
in the config is used to generate one python class.

* *Ontology* - A collection of `entries`, where each entry is represented as a class. 
The entries of an ontology can span one or more Python modules. This way, the entry classes
can be imported regardless of the ontology config they were originally generated from.
The modules that contain these entry classes generally belong to the package `ft.onto`.

* *Attribute* - An attribute generally corresponds to a label or property 
associated with an entry, like, `pos_tag` for the entry `Token`.

* *Top Entry* - Top entries are a set of entries that are pre-defined in the Forte library, 
in the module `forte.data.ontology.base.top`. All user-defined entries should extend one of 
the top entries. 
 
We provide a set of commonly used entries in the module ``forte.data.ontology.base_ontology.py``. 
Those entries could be reused directly and need not be redefined in the ontology config by the user.
 
### A simple ontology config ###
Let us define a simple ontology, that is used to define the NLP concepts, `Word` and `Phrase`.
```json
{
    "name": "simple_ontology",
    "description": "A simple Ontology",
    "definitions": [
        {
            "entry_name": "ft.onto.simple_ontology.Word",
            "parent_entry": "forte.data.ontology.top.Annotation",
            "attributes": [
                {
                    "name": "token_length",
                    "type": "int"
                }
            ]
        },
        {
            "entry_name": "ft.onto.simple_ontology.Phrase",
            "parent_entry": "forte.data.ontology.top.Annotation",
            "description": "Entry defining a phrase in the document.",
            "attributes": [
                {
                    "name": "token_length",
                    "type": "int"
                },
                {
                    "name": "phrase_tags",
                    "description": "To demonstrate the composite type, List.",
                    "type": "List",
                    "item_type": "str"
                }
            ]
        }
    ]
}
```
#### Breakdown of the simple ontology ####

- The top level `name` and `description` are annotation keywords meant for descriptive
purposes only.

- The `definitions` is used to enlist entry definitions, where each entry is represented
as a json object. For each entry, one class will be generated. The keywords 
of an entry definition are explained in the next section.

##### ```definitions``` #####
These are the commonly used fields for each entry class definition - 
* The `entry_name` keyword defines the name of the entry. It is used to 
define the full name of an entry, and is of the form
```<package_name><module_name>.<entry_name>```.
    * The package name is generally `ft.onto`. It is used to create the 
    package directory tree in which the generated module resides.
    * The `<module_name>` is the name of the generated file in which the entry 
    would be placed.
    > Note: Entries defined in the same config can have module names that are 
    different from each other.
    * The `<entry_name>` is used as the generated class name.
    
 * The `parent_type` keyword defines the base class of the generated entry class. All 
 the user-defined entries should inherit either any of the top entries or one 
 of the other user-defined entries.
 
 * The `description` keyword is optionally used as the comment to describe the generated Python class.
  
 * `attributes`: List of attributes that would be used as instance variables of 
 the generated class. Each keyword for an attribute is defined in the next 
 section.

##### ```attributes``` #####
These are the commonly used fields for each attribute of the entry definition - 
* The `name` keyword defines the name of the property unique to the entry.

* The `description` keyword is optionally used as the comment to describe the attribute.

* The `type` keyword is used to define the type of the attribute. Currently supported types are:
    * Primitive types - `int`, `float`, `str`, `bool`
    * Composite types - `List`
    * Entries defined in the `top` module - The attributes can be of the type base
    entries (defined in the `forte.data.ontology.top` module) and can be directly 
    referred by the class name.
    * User-defined types - The attributes can be of the type of entries that are
     user-defined. These user-defined entries could be defined (a) in the same config 
     (b) any of the imported configs. To avoid ambiguity, only full-names of the user-defined
     entry types are supported
     
* `item_type: str`: If the `type` of the property is one of the composite types,
 then `item_type` defines the type of the items contained in the property. 
As of now, we only support arrays of uniform types.

### Additional fields ###
The additional functionalities and fields provided by the ontology generation
framework are demonstrated through the following `example_complete_ontology`.
#### A complete ontology ####
```json
{
  "ontology_name": "example_complete_ontology",
  "import_paths": [
    "example_import_ontology_config.json"
  ],
  "additional_prefixes": [
    "custom.user"
  ],
  "definitions": [
    {
      "entry_name": "ft.onto.ft_module.Word",
      "parent_entry": "ft.onto.example_import_ontology.Token",
      "attributes": [
        {
          "name": "lemma",
          "type": "str"
        },
        {
          "name": "is_verb",
          "type": "bool"
        }
      ]
    },
    {
      "entry_name": "ft.onto.ft_module.Dependency",
      "parent_entry": "forte.data.ontology.top.Link",
      "attributes": [
        {
          "name": "rel_type",
          "type": "str"
        }
      ],
      "parent_type": "ft.onto.ft_module.Token",
      "child_type": "ft.onto.ft_module.Token"
    },
    {
      "entry_name": "custom.user.custom_module.Sentence",
      "parent_entry": "forte.data.ontology.top.Annotation",
      "attributes": [
        {
          "name": "words",
          "type": "List",
          "item_type": "ft.onto.ft_module.Word"
        }
      ]
    }
  ]
}
```
##### Additional top level fields #####
- The `additional_prefixes` keyword is optionally used to define a list
 of package names. This list is specified when the user prefers to use one or more
  custom package names for the generated entries instead of ``ft.onto``.
  
  For example, in the above ontology, the first entry ``custom.user.custom_module.Sentence``, will be generated in the package directory, ``custom.user`` which
  should be provided as a prefix in the ``additional_prefixes`` list.

- `import_paths` is an optional keyword used to define a list of json
config paths that the current config might depend on.
    - The entries of the imported configs can be used in the current config as
    types or parent classes.
    - The import paths could either be (a) absolute paths or 
    (b) relative to the directory of the current config or the
    current working directory (c) relative to one of the user-provided 
    ``config_paths`` (see [usage](#usage).)
    
    > For example, ``ft.onto.ft_module.Word`` has the parent entry defined in the
    generated module ``ft.onto.example_import_ontology``. The generation 
    framework makes sure that the imported JSON configs are generated before the
    current config. In case of cycle dependency between the JSON configs, an 
    error would be thrown.
    
##### Additional entry definition fields #####
* If the `parent_entry` is an instance of the type `forte.data.ontology.top.BaseLink`, two 
 additional fields can be defined in the entry definition - ``parent_type`` and ``child_type``. 
 These fields are used for strict type checking of parent link type and the child link type.
 
 For example, in the above ontology, the entry ``ft.onto.ft_module.Dependency``
    has the ``parent_entry`` and ``child_entry`` specified, as the parent is of 
    the type ``Link``.
    
* Similarly, if the `parent_entry` is an instance of type `forte.data.ontology.top.BaseGroup`, 
an additional field, ``member_type``, can be defined in the entry definition. The value of 
`member_type` is used for strict type checking of members of an entry of group type.

### Usage ###
* Write the json config(s) as per the instructions in the previous sections. 
* Use the command `generate_ontology --create` (added during installation of Forte) to 
generate the ontology, and `generate_ontology --clean` to clean up the generated ontology. 
The steps are detailed in the following sections.

 #### Generating the ontology ####
 Use ``create`` mode to generate ontology modules, along with their package directories
  given a root JSON config. If ontology generation is successful, the path of the directory 
  where the ontology is generated is printed.
 
 If ``--no_dry_run`` is not explicitly specified, the ontology package tree 
 is generated in a temporary directory. Otherwise it is generated in the value 
 of ``--dest_path`` if provided, or, current working directory if ``--dest_path`` 
 is not provided.
 
 ##### Ontology Generation Steps #####
Let us try to generate the `example_complete_ontology` as defined above. 

* Make sure that Forte is installed. To verify that the `generate_ontology`
command is found, run `generate_ontology -h`, and the output should look like the
following -
    ```bash
    $ generate_ontology -h
    usage: generate_ontology [-h] {create, clean} ...
    
    Utility to automatically generate or create ontologies.
    
    *create*: Generate ontology given a root JSON config.
    Example: python generate_ontology.py create --config forte/data/ontology/configs/example_ontology_config.json --no_dry_run
    
    *clean*: Clean a folder of generated ontologies.
    Example: python generate_ontology.py clean --dir generated-files
    
    positional arguments:
      {create, clean}
    
    optional arguments:
      -h, --help      show this help message and exit
    ```

* Create a user project directory and a config directory to hold the configs 
`example_complete_ontology_config.json` and `example_import_ontology_config.json`.
    ```bash
    $ tree user_project
    user_project
    └── configs
        ├── example_complete_ontology_config.json
        └── example_import_ontology_config.json
    
    1 directory, 2 files
    ```

* Let the imported json config, ``example_import_ontology_config.json`` contain the following content -
    ```json
    {
    "name": "example_import_ontology",
    "definitions": [
        {
            "entry_name": "ft.onto.example_import_ontology.Token",
            "parent_entry": "forte.data.ontology.top.Annotation",
            "description": "Base parent token entry",
            "attributes": [
                {
                    "name": "pos",
                    "type": "str"
                },
                {
                    "name": "lemma",
                    "type": "str"
                }
            ]
        },
        {
            "entry_name": "ft.onto.example_import_ontology.EntityMention",
            "parent_entry": "forte.data.ontology.top.Annotation",
            "attributes":
            [
                {
                    "name":"entity_type",
                    "type": "str"
                }
            ]
        }
    ]
    }
    ```

* The next steps is to generate the ontology. All the arguments of `generate_ontology create` are explained as below: 
     ```bash
     usage: generate_ontology create [-h] --config CONFIG [--no_dry_run]
                                    [--dest_path DEST_PATH]
                                    [--config_paths [CONFIG_PATHS [CONFIG_PATHS ...]]]
    
    optional arguments:
      -h, --help            Show this help message and exit.
      
      --config CONFIG       Root JSON config.
      
      --no_dry_run          Generates the package tree in a temporary directory if
                            true, ignores the argument `--dest_path`
                            
      --dest_path DEST_PATH
                            Destination directory provided by the user. Only used
                            when `--no_dry_run` is specified. The default directory
                            destination directory is the current working directory.
                            
      --config_paths [CONFIG_PATHS [CONFIG_PATHS ...]]
                            Paths in which the root and imported config files are
                            to be searched.
     ```

* Run the `generate_ontology` command in `create` mode. Let the destination path be a directory `src` in user project.

    ```bash
    $ generate_ontology create --config configs/example_complete_ontology_config.json --dest_path src
       
    INFO:scripts.generate_ontology:Ontology will be generated in a temporary directory as --no_dry_run is not specified by the user.
    INFO:scripts.generate_ontology:Ontology generated in the directory /var/folders/kw/ffm8tdn57vg3msn7fhr2htpw0000gp/T/tmpc1kp_xdp.
    ```

* Note that the ``--dest_dir`` is not used at all and the data is generated in a temporary directory, as, by default, ``generate_ontology`` runs in the ``dry_run`` mode. To use the value of `--dest_path`, `--no_dry_run` has to be passed.
    ```bash
    $ generate_ontology create --config configs/example_complete_ontology_config.json --dest_path src --no_dry_run
    INFO:scripts.generate_ontology:Ontology generated in the directory /Users/mansi.gupta/user_project/src
    ```

 * Let's look at the generated file structure.
     ```bash
    tree user_project
    .
    ├── configs
    │   ├── example_complete_ontology_config.json
    │   └── example_import_ontology_config.json
    └── src
        ├── custom
        │   └── user
        │       ├── .generated
        │       └── custom_module.py
        └── ft
            └── onto
                ├── example_import_ontology.py
                └── ft_module.py
    
    6 directories, 9 files
    ```
 * Our ontology generation is complete!
 
 #### Cleaning the generated ontology ####
* Use `clean` mode of `generate_ontology` to clean the generated files from a given directory.
* All the arguments of `generate_ontology clean` are explained as below:

 ```bash
usage: generate_ontology clean [-h] --dir DIR [--force]

optional arguments:
  -h, --help  show this help message and exit
  --dir DIR   Generated files to be cleaned from the directory path.
  --force     If true, skips the interactive deleting offolders. Use with
              caution.
 ```
 
* Now, let's try to clean up *only* the automatically generated files and directories. 
Say, there are user-created files in the generated folder, ``user_project/src/ft``, 
called `important_stuff` that we do not want to clean up. 
     ```bash
  $ mkdir user_project/src/ft/important_stuff
  $ touch user_project/src/ft/important_stuff/file.py
  $ tree user_project
    .
    ├── configs
    │   ├── example_complete_ontology_config.json
    │   └── example_import_ontology_config.json
    └── src
        ├── custom
        │   └── user
        │       └── custom_module.py
        └── ft
            ├── important_stuff
            │   └── file.py
            └── onto
                ├── example_import_ontology.py
                └── ft_module.py
    ``` 

* Run the cleanup command and observe the directory structure. The cleanup preserves the 
partial directory structure in the case there exists files that are not generated by the framework.
    ```bash
    $ generate_ontology clean --dir user-project/src
  
    INFO:scripts.generate_ontology.__main__:Directory /Users/mansi.gupta/user_project/src not empty, cannot delete completely.
    INFO:scripts.generate_ontology.__main__:Deleted files moved to /Users/mansi.gupta/user_project/.deleted/2019-11-28-02-53-29-952561.
    ```
    
* For safety, the deleted directories are not immediately deleted but are moved to a timestamped 
 directory inside ``.deleted`` and can be restored, unless `--force` is passed. 
 
 * If the directories that are to be generated already exist, the files will be generated in the 
 already existing directories.
 
 * Automatically generated folders are identified by an empty marker file of the name ``.generated``, 
 and automatically generated files are identified by special headers. If the headers or marker files 
 are removed manually, than the cleanup won't affect them.
 
