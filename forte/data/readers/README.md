Reader readme
This document provides a overview of how to implement reader (raw data processor).

### _collect:  file_path ->  takes input and yields data (actually returns an iterator)
Takes a file path and returns an iterator

### _parse_pack: another function returns an iterator
Takes a row from collect iterator and pack data into `DataPack` and returns an `DataPack Iterator`
#### General procedure
1. create a `DataPack` and `set_text`
2. pass `DataPack` into different ontologies to create special class variables/methods. 
   Most commonly class variables are start indices and end indices for the dataclass. 


### _cache_key_function: 
