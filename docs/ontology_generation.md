# Building and Understanding Ontology

Forte is built on top of an _Ontology_ system, which defines the relations
between NLP annotations, for example, the relation between words and documents,
or between two words. This is the core for Forte.

The ontology can be specified via a JSON format. And
tools are provided to convert the ontology into production code (Python). 
Make sure Forte is installed before following this tutorial.

# A simple ontology config
Imagine you need to develop an NLP system for a pet shop, first thing first, 
you need to understand what are the needed output from the documents. Let's 
say you need to develop a system to assets such as `Pet` and `Revenue` ,
and hopefully automatically find out these from text. We have built an example
ontology here: [pet shop ontology](https://github.com/asyml/forte/blob/master/examples/ontology/pet_shop.json)

Now, before we go to the details, at Forte's root directory, try run the following command:
```shell
generate_ontology create -i examples/ontology/pet_shop.json -o examples/ontology -r
```
If run successfully, you will find some python code being generated in `examples/ontology`,
under the package `awesome.pet.com`. This is what the Forte ontology system does, it generates
the python classes needed to handle the NLP data structures.

The JSON ontology spec should be quite self-explanatory, we define types like `Pet` and `Owner`,
which have some attributes. And the `Owner` have a list of `Pet`. The python code
exactly represent the structure.

In the rest of the tutorial, we will walk through this example and you will learn:
  * Define a simple ontology spec for your project.
  * Import other ontology(s) to build yours.
  * Generate the corresponding Python classes automatically and use them in 
    your project.

# Before we start
There are a few basic concepts to understand Forte's ontology system.

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

* *Top* - Top entries are a set of entries that are pre-defined in the Forte library, 
in the module `forte.data.ontology.base.top`. All user-defined entries should extend one of 
the top entries. 
 
We provide a set of commonly used NLP entry types in the module 
[``forte.data.ontology.base_ontology.py``](https://github.com/asyml/forte/blob/master/forte/ontology_specs/base_ontology.json). 
Those entries could be used directly in your project!
 
# A simple ontology config
Let us consider a simple ontology for documents of a pet shop.
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
            "entry_name": "ft.onto.pet_shop.Owner",
            "parent_entry": "forte.data.ontology.top.Annotation",
            "description": "Owner of pets.",
            "attributes": [
                {
                    "name": "name",
                    "type": "str"
                },
                {
                    "name": "pets",
                    "description": "List of pets the owner have.",
                    "type": "List",
                    "item_type": "ft.onto.pet_shop.Pet"
                }
            ]
        }
    ]
}
```

## Breakdown of the simple ontology

- The top level `name` and `description` are annotation keywords meant for descriptive
purposes only.

- The `definitions` is used to enlist entry definitions, where each entry is represented
as a json object. Each entry correspond to one concept in the ontology, and a Python class. 

### ```definitions```
Each definition is a dictionary of several keywords:
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
 the generated class. 

### ```attributes```
Each entry definition will define a couple (can be empty) attributes, mimicking the class variables:

* The `name` keyword defines the name of the property unique to the entry.
* The `description` keyword is optionally used as the comment to describe the attribute.
* The `type` keyword is used to define the type of the attribute. Currently supported types are:
    * Primitive types - `int`, `float`, `str`, `bool`
    * Composite types - `List`, `Dict`
    * Entries defined in the `top` module - The attributes can be of the type base
    entries (defined in the `forte.data.ontology.top` module) and can be directly 
    referred by the class name.
    * User-defined types - The attributes can be of the type of entries that are
     user-defined. These user-defined entries could be defined (a) in the same config 
     (b) any of the imported configs. To avoid ambiguity, only full-names of the user-defined
     entry types are supported     
* `item_type: str`: If the `type` of the property is a `List`,
   then `item_type` defines the type of the items contained in the list. 
* `key_type` and `value_type`: If the `type` of the property is a `Dict`,
   then these two represent the types of the key and value of the dictionary,
   currently, only primitive types are supported as the `key_type`.

# Major ontology types, Annotations, Links, Groups and Generics
There are some very frequently used types in NLP: 

* **Annotation**: an annotation is a type of entry that correspond to a piece of text.
  For example, a `sentence` can be an annotation. In our example, we use
  `awesome.pet.com.Color` to annotate the color words in text documents. All
  annotations need to inherit `forte.data.ontology.top.Annotation`. The annotation 
  entries will have special `begin` and `end` attributes to indicate their text 
  position.

* **Link**: a link is a type of entry that connect two other entries. For example, a dependency
  link connect two words. All link in Forte need to inherit `forte.data.ontology.top.BaseLink`,
  and the ontology need to specify ``parent_type`` and ``child_type`` for the linked entries. 
    
* **Group**: a group is a type of entry that groups several entries. For example, a coreference
  cluster will contain several entity mentions. All link in Forte inherits from
  `forte.data.ontology.top.BaseGroup`.  The ``member_type`` need to be set to indicate the
  types of entries in the group.
  
* **Generics**: there are some entries that do not have the above characteristics, such as
  general meta data storing information. These are `Generics` types. 

To see more examples of these different types of entries, you can read the 
[pet shop ontology](https://github.com/asyml/forte/blob/master/examples/ontology/pet_shop.json)
as an example, or refer to the [base ontology](https://github.com/asyml/forte/blob/master/forte/ontology_specs/base_ontology.json),
which is an ontology provided by Forte to represent general NLP concepts.

# Importing another ontology
`imports` is an optional keyword used to help you import existing ontology to help build
the current one. This is similar to `import` in a normal programming language:

* The entries of the imported configs can be used in the current config as
types or parent classes.
* The imports could either be 
  - absolute paths
  - relative to the directory of the current config or the
current working directory
  - relative to one of the user-provided ``spec_paths`` (see [generation steps](#ontology-generation-steps).)
    
For example, ``ft.onto.ft_module.Word`` has the parent entry defined in the
generated module ``ft.onto.example_import_ontology``. The generation 
framework makes sure that the imported JSON configs are generated before the
current config. In case of cycle dependency between the JSON configs, an 
error would be thrown.

# Package Naming Convention
Each entry should be named following a package convention, such as `ft.onto.Pet`
in this example. This allows the generator to create a package structure for the python
class.

In order to avoid polluting your package space accidentally, the package names
that can be used on the types are restricted. The default package name allowed
is `ft.onto`. However, in many cases you may want to use custom package name,
 let's say,`awesome.pet.com`, how can we achieve it?
  
We can explicitly set more attributes in the `additional_prefixes`, as in the 
following snippet:

```json
{
  "name": "pet_shop_ontology",
  "additional_prefixes": [
    "awesome.pet.com"
  ],
  "description": "An Ontology Used to manage the pet shop assets and pets",
  "definitions": [
    {
      "entry_name": "awesome.pet.com.Color",
      "parent_entry": "forte.data.ontology.top.Annotation",
      "description": "Annotation for color words.",
      "attributes": [
        {
          "name": "color_name",
          "type": "str"
        }
      ]
    },
    {
      "entry_name": "awesome.pet.com.Pet",
      "parent_entry": "forte.data.ontology.top.Annotation",
      "description": "Pets in the shop.",
      "attributes": [
        {
          "name": "pet_type",
          "type": "str"
        },
        {
          "name": "color",
          "type": "awesome.pet.com.Color"
        }
      ]
    }
  ]
}
```
    
# Generating Python classes from ontology.
* Write the json spec(s) as per the instructions in the previous sections. 
* Use the command `generate_ontology --create` (added during installation of Forte) to 
generate the ontology, and `generate_ontology --clean` to clean up the generated ontology. 
The steps are detailed in the following sections.

## Ontology Generation Steps
At the beginning we have tried generating the ontology. Now let's go into the
some details.

* To verify that the `generate_ontology`
command is found, run `generate_ontology -h`, and the output should look like the
following -
    ```
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

* All the arguments of `generate_ontology create` are explained as below: 
     ```
    usage: generate_ontology create [-h] -i SPEC [-r] [-o DEST_PATH]
                                    [-s [SPEC_PATHS [SPEC_PATHS ...]]]
                                    [-m MERGED_PATH] [-e] [-a]
    
    optional arguments:
      -h, --help            show this help message and exit
      -i SPEC, --spec SPEC  The main input JSON specification.
      -r, --no_dry_run      Generates the package tree in a temporary directory if
                            true, ignores the argument `--dest_path`
      -o DEST_PATH, --dest_path DEST_PATH
                            Destination directory provided by the user. Only used
                            when --no_dry_run is specified. The default directory
                            is the current working directory.
      -s [SPEC_PATHS [SPEC_PATHS ...]], --spec_paths [SPEC_PATHS [SPEC_PATHS ...]]
                            Paths in which the root and imported spec files are to
                            be searched.
      -m MERGED_PATH, --merged_path MERGED_PATH
                            The destination file path for the mergedfile path.
      -e, --exclude_init    Excludes generation of `__init__.py` files in the
                            already existing directories, if`__init__.py` not
                            already present.
      -a, --gen_all         If True, will generate all the ontology,including the
                            existing ones shipped with Forte.

     ```

* Run the `generate_ontology` command in `create` mode. Let the destination path be a directory `src` in user project.

    ```bash
    $ generate_ontology create -i configs/example_complete_ontology_config.json -o src
       
    INFO:scripts.generate_ontology:Ontology will be generated in a temporary directory as -r is not specified by the user.
    INFO:scripts.generate_ontology:Ontology generated in the directory /var/folders/kw/ffm8tdn57vg3msn7fhr2htpw0000gp/T/tmpc1kp_xdp.
    ```

* Note that the ``-o`` is not used at all and the data is generated in a temporary directory, as, by default, ``generate_ontology`` runs in the ``dry_run`` mode. To use the value of `--dest_path`, `--no_dry_run` has to be passed.
    ```bash
    $ generate_ontology create -i configs/example_complete_ontology_config.json -o src -r
    INFO:scripts.generate_ontology:Ontology generated in the directory /Users/mansi.gupta/user_project/src
    ```

 * Let's look at the generated file structure.
     ```bash
    tree awesome
    
    awesome
    ├── __init__.py
    └── pet
        ├── com.py
        └── __init__.py
    ```
 * Our ontology generation is complete!
 
## Cleaning the generated ontology
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
 
