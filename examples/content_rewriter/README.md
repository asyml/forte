## A Content Re-writer Example

This example implements a content rewriter. Given a table and a sentence, this
will rewrite the sentence based on the table.

### Prerequisites

The code has been tested on:
 - Python 3.6.0 and Python 3.7.6
 - tensorflow-gpu==1.14.0
 - texar-pytorch==0.1.1
 - texar==0.2.1
 - cuda 10.0

** NOTE **: 
Due to some historical texar compatibility issue, the model is only compatible
by installing texar 0.2.1 from source, which can be installed via the following
command.

```bash
wget https://github.com/asyml/texar/archive/v0.2.1.zip
cd texar && pip install .
```

Run the following commands:

```bash
cd model
pip install -r requirements.txt
``` 

### Downloading the models and data

Before we run the rewriting demo, we need to download models and data from the 
[link](https://drive.google.com/drive/folders/1jNaJ_R_f89G8xbAC8iwe49Yx_Z-LXr0i?usp=sharing) 
and put the two directories(e2e_data, e2e_model) under the same directory [model_dir]

### Running the example

Now to see the example in action, just run

```bash
python pipline.py [model_dir]
```

### Using in Stave Front-End

We have developed a simple Front End for this using [stave](https://github.com/asyml/stave/blob/master/src/plugins/dialogue_box/READEME.md). You can use the data in `table_inputs` folder to play with the bot. If you would like to regenerate the preprocessing data, just run

```bash
python prepare_pipele.py
```
