# DataPack Tutorial

## Get primitive data
`DataPack.get_data()` is commonly used to retrieve primitive data from a datapack. User can request data of a certain `Annotation` type (currently supporting `Annotation` and `AudioAnnotation`) by setting parameter `context_type`. User can also request data for certain data fields.

For example, if User wants to get primitive data of `AudioAnnotation` from a `DataPack` instance `pack`. User can call the function like the code blow. It returns a generator that User can iterate over.
`AudioAnnotation` is passed into the method as parameter `context_type`.
```python
pack.get_data(AudioAnnotation)
```

For example, if User wants to get primitive data of `AudioAnnotation` from a `DataPack` instance `pack` and specific data fields such as `recording_class` for `Recording` entry and `speaker` for `AudioUtterance` entry. User can call the function like the code blow.
`Recording` and `AudioUtterance` are subclass of `AudioAnnotation`. Therefore, their data fields can be requested when `context_type` is `AudioAnnotation`. Since they have data fields that are different. User needs to let dictionary keys be those subclass of requested `context_type` and values be requested data fields in the corresponding subclass.
```python
pack.get_data(AudioAnnotation,
                {Recording:
                    {"fields": ["recording_class"]},
                AudioUtterance:
                    {"fields": ["speaker"]}}
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
