# TODO Notes
1. A name for nlp_pipeline, anyone?
    1. Assigned to everyone
1. Prepare config system.
    1. Option class and YAML parse: zecong
1. Resource Manager.
    1. basically a dict to references
    1. add some register function 
        1. ResourceManager.get(resource_name)
        1. ResourceManager.put(resource_name, resource)
        1. zecong
1. Reorg initialize and init: weiwei
1. ~~Ontology management: weiwei~~
    1. ~~When there are many ontologies, how can we simplify them?~~
    1. ~~How should we store them if the ontologies are all over the place.~~
1. ~~Batching support.~~
    1. ~~Add a customizable sharable Batching interface.~~
1. ~~The process() function in Trainer need a name and a purpose.~~
    1. ~~Haoran gives a new name: consume()~~
1. ~~Logging clean up.~~
    1. ~~Delete logging config.~~
1. ~~Accessing problem:~~
    1. ~~Many fields should be stored as private, such as text, annotations in data_pack~~
    1. ~~Use getter, setter and add_xxx function instead of directly manipulating the fields.~~
1. Reduce the interchange format size if possible?
    1. ~~Probably no need to store all indexing along with it~~
1. Prepare to integrate with Executor.
1. ~~Some pipeline usage problems:~~
    1. ~~Use the with statement to simplify the load anv verbosity in processors~~
    1. ~~Reduce the need to calling "__name__" for users~~
    1. ~~Now the way to create Link requires a user to pass in the "tid", which add
     burden to the users to maintain such an id.~~
1. There is no component in set_field(), might be confusing.
1. Discussion of the Multipack interface.
1. How about the "view" method?
    1. Can be used to deal with the train test problem
    1. Can be used to do ensemble: the center distribute the views to different 
    classes, and it will take them back for ensemble.

## Symphony Problem:
1. Make sure the pipeline is stateless, so that in symphony
 you can start a new pipeline all the time.
1. Let's try to have a JSON schema in the future.
    1. Useful for validation
1. Let the med team use the non-symphony version.
1. Is it possible to ignore entries in JSON that the ontology that it doesn't know
