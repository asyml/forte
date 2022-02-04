# DataPack Tutorial

## Get primitive data
`DataPack.get_data()` is commonly used to retrieve primitive data from a `DataPack`. User can request particular data fields within the range of a particular `Annotation` or `AudioAnnotation` type. User request particular data fields by setting `request` and the search range by setting `context_type`.

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

For example, if User wants to get primitive data of `AudioAnnotation` from a `DataPack` instance `pack`. User can call the function like the code blow. It returns a generator that User can iterate over.
`AudioAnnotation` is passed into the method as parameter `context_type`.
```python
pack.get_data(AudioAnnotation)
```

For example, if User wants to get primitive data of `AudioUtterance` from a `DataPack` instance `pack` and specific data fields such as `speaker` for `AudioUtterance` entry. User can call the function like the code blow.

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
