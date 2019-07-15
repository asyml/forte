# TODO Notes
1. Prepare config system.
    1. How to join well with Texar.
1. Batching support.
    1. Add a customizable sharable Batching interface.
1. A name for nlp_pipeline, anyone?
1. Some changes for Pack to tensor conversion.
    1. ~~Always return the full offsets.~~
    1. ~~When requested, return additional token and context based (or other)
     offsets~~
1. Trainer, Processor separation:
    1. Trainer no need to extent Processor
1. ~~Predictor revisit~~
    1. ~~Now no one is using the Process() function directly, some changes here~~
1. Resource Manager.
1. Logging clean up.
1. Assessing problem:
    1. Many fields should be stored as private, such as text, annotations in data_pack
    1. Use getter, setter and add_xxx function instead of directly manipulating the fields.
1. Reduce the interchange format size if possible?
    1. Probably no need to store all indexing along with it
1. Prepare to integrate with Executor.
1. Some simplification measures:
    1. Use the with statement to simplify the load anv verbosity in processors
    1. Reduce the need to calling "__name__" for users
1. There is no component in add_field(), might be confusing.
1. Discussion of the Multipack interface.
1. How about the "view" method?
    1. Can be used to deal with the train test problem
    1. Can be used to do ensemble: the center distribute the views to different 
    classes, and it will take them back for ensemble.
