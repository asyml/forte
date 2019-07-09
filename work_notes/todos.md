# TODO Notes
1. Prepare config system.
1. Batching support.
    1. Add a customized sharable Batching interface.
1. Some changes for Pack to tensor conversion.
    1. Always return the full offsets.
    1. When requested, return additional token and context based (or other)
     offsets
1. Trainer, Processor separation:
    1. Trainer no need to extent Processor
1. Predictor revisit
    1. Now no one is using the Process() function directly, some changes here
1. Resource Manager.
1. Other code clean up.
1. Logging clean up.
1. Reduce the interchange format verbosity if possible?
1. Prepare to integrate with Executor.
