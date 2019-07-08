"""
In this pipeline, we need to
1.  Build data packs from multiple data sets
2.  create a vocabulary over the training data packs (With processor)
3.  create a model processor
4.  Use trainer processor to perform training
        trainer only accepts data and
5. Use validation processor on validation dataset
6.

The training logic (where to save the model, update learning rate) is held
    in pipeline?
"""