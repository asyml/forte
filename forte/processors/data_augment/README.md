# Data Augmentation

-----------------

Forte implements the interfaces and a few very useful data augmentation algorithms.

## Base Data Augmentation Processor and Replacement Ops

We provide a base data augmentation processor that supports common data operations, such
as text replacement, insertion and deletion. The processor provides interfaces for the
three types of operation and track them to update the other relevant entries (e.g.
update the dependency link after swapping a word).

Based on this processor, we can plugin a few based algorithms (named Replacment Op),
which are implemented in the `forte.processors.data_augment.algorithms` package.

### Dictionary Replacement Op

This Op replaces the input word with its synonym, antonym, hypernym or hyponym, based on
WordNet.

### Distribution Replacement Op

This Op replaces the input word by sampling from a probability distribution over a
vocabulary, such as uniform distribution, or unigram distribution. Users can implement
custom distributions.

### Embedding Replacement Op

This Op replaces the input with word embeddings. With a set of word embeddings, it will
search for similar words by calculating the cosine similarity between the embeddings,
and return the top-k similar ones. The Glove embeddings are provided by default, and
users can load their own word embeddings for the replacement.

### Back Translation Op

This Op utilizes machine translation for sentence replacement. It translates the input
from source language to target language, then translates it back to the source language.
Under the assumption of decent translators, the output should have a similar semantic
meaning as the input sentence, but possibly in a different presentation. We wrapped the
Marian MT as the translator which comes with customized language pairs. Customized
translators can be also implemented following our interface.

### Typo Replacement Op

This Op generates a typo by employing a typo replacement dictionary to replace a word with a relevant typo. 
It uses a pre-defined spelling mistake dictionary to simulate spelling mistakes.

## Easy Data Augmentation (EDA)

The Easy Data Augmentation (EDA) is a set of simple random text transformations, which
are supported by the Replacement Ops mentioned above. The original EDA paper proposes 4
different operations: Synonym Replacement (SR), Random Insertion (RI), Random Swap (RS)
and Random Deletion (RD). They are implemented as Processors in
`forte.processors.data_augment.algorithms.eda_processors`.

References:

```
Jason Wei and Kai Zou. Eda: Easy data augmentation techniques
for boosting performance on text classification tasks. arXiv preprint
arXiv:1901.11196, 2019.
```

## Unsupervised Data Augmentation (UDA)

The Unsupervised Data Augmentation(UDA) can utilize unsupervised data and incorporate
the unsupervised loss function into supervised loss. We wrap the UDA consistency loss
and implement a UDA based data iterator at
`forte.processors.data_augment.algorithms.UDAIterator`

References:

```
Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, and Quoc V Le.
Unsupervised data augmentation for consistency training. arXiv preprint
arXiv:1904.12848, 2019.
`````

## Reinforcement Based Data Augmentation

This algorithm uses reinforcement learning technique to jointly learn the data
augmentation model and train the downstream mode. We provide a wrapper class to convert
an existing model to use this manipulation technique. For details on how to use this,
check out
our [example](https://github.com/asyml/forte/tree/master/examples/data_augmentation/reinforcement)
here.

References:

```
Zhiting Hu, Bowen Tan, Ruslan Salakhutdinov, Tom Mitchell, and Eric P.
Xing. Learning Data Manipulation for Augmentation and Weighting. In
NeurIPS 2019, 2019.
```

## More Information

For more information of our data augmentation modules, read
our [capstone project report](https://github.com/asyml/forte/blob/master/docs/reports/Capstone_Data_Augmentation-2.pdf)
.
