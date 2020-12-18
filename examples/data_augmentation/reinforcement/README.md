## Reinforcement Learning based Data Augmentation for Text Classification

This model builds upon the connection of supervised learning and reinforcement learning (RL), and adapts reward learning algorithm from RL for joint data augmentation learning and model training.  
For details, please refer to the [Learning Data Manipulation for Augmentation and Weighting](https://arxiv.org/pdf/1910.12795.pdf) and the original [github](https://github.com/tanyuqian/learning-data-manipulation/).

This example demonstrates the usage of `forte/models/da_rl/MetaAugmentationWrapper`, that wraps a BERT Masked Language Model data augmentation model to perform this RL adaptive learning with a BERT-based text classifier downstream model.


### Examplge Usage

See `RLAugmentClassifierTrainer.train_epoch()` in `main.py` for details.

```python

# aug_wrapper: MetaAugmentationWrapper
# model: downstream BERT model
# optim: optimizer associated with model

for batch in training_data:
    # Train augmentation model params phi.
    aug_wrapper.reset_model()
    # Iterate over training instances.
    for instance in batch:
        bert_features = instance

        # Augmented instance with params phi exposed
        aug_bert_features = aug_wrapper.augment_instance(bert_features)

        # Compute downstream model loss.
        model.zero_grad()
        loss = model(aug_bert_features)
        # Update classifier params on meta_model.
        meta_model = MetaModel(model)
        meta_model = aug_wrapper.update_meta_model(
            meta_model, loss, model, optim)

        # Compute grads of aug_model on validation data.
        for val_batch in validation_data:
            val_bert_features = val_batch
            val_loss = meta_model(val_bert_features)
            val_loss = val_loss / len(batch) / args.num_aug / len(validation_data)
            val_loss.backward()

    # Update aug_model param phi.
    aug_wrapper.update_phi()

    # Train classifier with augmented batch
    batch_bert_features = batch
    aug_bert_features = aug_wrapper.augment_batch(batch_bert_features)

    optim.zero_grad()
    loss = model(aug_bert_features)
    loss.backward()
    optim.step()
```


### Train

```bash
python main.py --do-train --do-eval --do-test
```

To change the classifier hyperparameters, please see `config/config_data.py`.


## Results

Results running from this example align with the results provided in the [paper](https://arxiv.org/pdf/1910.12795.pdf), please refer to it for more details.  
Training on the IMDB dataset:  
Subsample a small training set of 40 instances for each class.  
Then further create a small validation set of 5 instances per class.  
Test set is of 25000 instances.

| Number of Training Instances | Accuracy |
| -------------------------- | ------------- |
| 80                        | 65.12         |
| 800                      | 75.67         |
| 8000                      | 82.23         |
| 15000                      | 83.33         |

You can further improve the performance by tuning hyperparameters.
