## A Clinical NER Example

This example implements a Clinical NER model. Given a clinical document, this example will annotate NER labels based on the document. This model is borrowed from [CliNER](https://github.com/text-machine-lab/CliNER).

### Prerequisites

```
cd CliNER && pip install requirements.txt
```

### Example Data


Although we cannot provide the data due to the license problem, there is a sample to demonstrate how the data is formatted (not actual data from i2b2, though).

    CliNER/data/examples/ex_doc.txt

This is a text file. Discharge summaries are written out in plaintext, just like this. It is paired with a concept file, which has its annotations.

    CliNER/data/examples/ex_doc.con

This is a concept file. It provides annotations for the concepts (problems, treatments, and tests) of the text file. The format is as follows - each instance of a concept has one line. The line shows the text span, the line number, token numbers of the span (delimited by white space), and the label of the concept.

Please note that the example data is simply one of many examples that can found online.


### Running with the example data

First, we need to download the pre-trained model as follows:

```bash
sh download_model.sh
```

Then we can prepare data as follows:

```bash
python prepare_pipline.py
```

To inference with the example data, just run:

```bash
python pipline.py predict
```
