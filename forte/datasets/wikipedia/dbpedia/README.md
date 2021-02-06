###Wikipedia based NLP datasets
This reader reads Wikipedia strcutures and info-boxes to create a dataset for
wiki-based NLP and text mining.

This reader is currently based on the DBpedia 2016/10 [dataset](https://wiki.dbpedia.org/downloads-2016-10),
and relies on the following specific data sources:
  1. NIF Context: this provides the full Wikipedia text.
  1. NIF page structure: this provides the detailed structure of the article.
  1. Mapping based Literals and Mapping based Objects: these two provide the
     info box information of each page.
  1. NIF text links: provides the linking between pages.
