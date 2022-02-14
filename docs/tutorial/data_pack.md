# DataPack Tutorial

## Get data
`DataPack.get_data()` is commonly used to retrieve data from a `DataPack`. This method returns a generator that generates dictionaries containing data requested, and each dictionary has a scope that covers certain range of data in the `DataPack`.
To understand this, let's consider a dummy case.  Given that there is a document in the `DataPack` instance `data_pack`, we want to get the full document in `data_pack`.
We can set up the `data_pack` using the following code under forte project root path.
```python
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
data_path = os.path.abspath(
            os.path.join("data_samples", "ontonotes/one_file"
            )
        )
pipeline: Pipeline = Pipeline()
pipeline.set_reader(OntonotesReader())
pipeline.initialize()
data_pack: DataPack = pipeline.process_one(data_path)
```


We can run the following code to get the full document.
```
for doc_idx, d in enumerate(data_pack.get_data(context_type=Document)):
    print(doc_idx, ":  ", d['context'])
```
Then the printed output is
```
0 :   The Indonesian billionaire James Riady has agreed to pay $ 8.5 million and plead guilty to illegally donating money for Bill Clinton 's 1992 presidential campaign . He admits he was trying to influence American policy on China .
```
As we can see, the generator generates a dictionary for each iteration and sentence data can be retrieved by dictionary key `'context'`.

To better understand this, let's consider a more concrete case. Since the document containing two sentences, suppose we want to retrieve text data sentence by sentence for a linguistic analysis tasks. In other words, we expect two dictionaries in the generator and each dictionary stores a sentence.

We can get each sentence by the following code
```python
data_generator = data_pack.get_data(context_type=Sentence)
for sent_idx, d in enumerate(data_generator):
    print(sent_idx, d['context'])
```
The printed output is
```
0 :   The Indonesian billionaire James Riady has agreed to pay $ 8.5 million and plead guilty to illegally donating money for Bill Clinton 's 1992 presidential campaign .
1 :   He admits he was trying to influence American policy on China .
```
As we can see, we get the two sentences by two iterations.

So far, we introduce two examples to explain the first parameter `context_type` which controls the granularity of the data context. Depending on the task, we can generate data of different granularities. We assigned `context_type` from `Document` to `Sentence` for sentence tasks, and we can even further change it to `Token` for token tasks.

Suppose we don't want to analyze the first sentence in the `data_pack`, there is `skip_k` parameter that skips k data of `context_type` and starts generating data from k+1 instance.

```python
data_generator = data_pack.get_data(context_type=Sentence, skip_k=0)
for sent_idx, d in enumerate(data_generator):
    print(sent_idx, d['context'])
```



 User can request particular data fields within the range of a particular `Annotation` or `AudioAnnotation` type. User request particular data fields by setting `request` and the search range by setting `context_type`.

### Annotation
In forte, each annotation has a range which includes begin and end of annotation-specific data of that particular annotation. For `Annotation` type, range means the begin index and end index of characters under `Annotation` type in the `text` payload of the `DataPack`.
```python
requests = {
    Sentence: ["speaker"],
    Token: ["pos", "sense"],
    PredicateMention: [],
    PredicateArgument: {"fields": [], "unit": "Token"},
    PredicateLink: {
        "component": utils.get_full_module_name(OntonotesReader),
        "fields": ["parent", "child", "arg_type"],
    },
}
pack.get_data(Annotation, requests)
```

 For an `Token` instance which is a subtype of `Annotation`, its annotation-specific data is `text` and therefore range means the begin and end of characters of that `Token` instance. For an `Recording` instance which is a subtype of `AudioAnnotation`, its annotation-specific data is `audio` and there range means the begin and end index of that `Recording` instance.





### AudioAnnotation
Based on the idea of "range", in the example code, entry `AudioUtterance` will be searched in `DataPack.audio_annotations` and the requested data field `speaker` will be included in the generator's data.

For `AudioAnnotation` type, range means the begin index and end index of sound sample under `AudioAnnotation` type in the `audio` payload of the `DataPack`.

For example, if User wants to get data of `AudioAnnotation` from a `DataPack` instance `pack`. User can call the function like the code blow. It returns a generator that User can iterate over.
`AudioAnnotation` is passed into the method as parameter `context_type`.
```python
pack.get_data(AudioAnnotation)
```

For example, if User wants to get data of `AudioUtterance` from a `DataPack` instance `pack` and specific data fields such as `speaker` for `AudioUtterance` entry. User can call the function like the code blow.

```python
pack.get_data(AudioAnnotation,
                {
                AudioUtterance:
                    {"fields": ["speaker"]}
                }
            )
```


## Build Coverage Index
`DataPack.get()` is commonly used to retrieve entries from a datapack. In some cases, we are only interested in getting entries from a specific range. `DataPack.get()` allows users to set `range_annotation` which controls the search area of the sub-types. If `DataPack.get()` is called frequently with queries related to the `range_annotation`, you may consider building the coverage index regarding the related entry types. Users can call `DataPack.build_coverage_for(context_type, covered_type)` in order to create a mapping between a pair of entry types and target entries that are covered in ranges specified by outer entries.

For example, if you need to get all the `Token`s from some `Sentence`, you can write your code as:
```python
# Iterate through all the sentences in the pack.
for sentence in input_pack.get(Sentence):
    # Take all tokens from a sentence
    token_entries = input_pack.get(
        entry_type=Token, range_annotation=sentence
    )
```
However, the snippet above may become a bottleneck if you have a lot of `Sentence` and `Token` entries inside the datapack. To speed up this process, you can build a coverage index first:
```python
# Build coverage index between `Token` and `Sentence`
input_pack.build_coverage_for(
    context_type=Sentence
    covered_type=Token
)
```
This `DataPack.build_coverage_for(context_type, covered_type)` function is able to build a mapping from `context_type` to `covered_type`, allowing faster retrieval of inner entries covered by outer entries inside the datapack.
We also provide a function called `DataPack.covers(context_entry, covered_entry)` for coverage checking. It returns `True` if the span of `covered_entry` is covered by the span of `context_entry`.
