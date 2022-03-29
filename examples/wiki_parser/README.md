# A Complex Reader Example

Sometimes the dataset is quite complex to read. Here, we provide a real example
where we use Forte to parse Wikipedia articles using Datasets from DBpedia.

```bash
python wiki_dump_parse.py [dbpedia datasets directory] [output directory]
```

There are quite a few datasets needed to be downloaded into the input folder. 
You can download them at 

Before that, you can try this example with the directory located at https://github.com/asyml/forte/tree/master/data_samples/dbpedia

```bash
python wiki_dump_parse.py ../data_samples/dbpedia output
```