# Running the example

## Install some wrappers for the example.

```bash
# Install huggingface
pip install "git+https://git@github.com/asyml/forte-wrappers.git#egg=forte.huggingface&subdirectory=src/huggingface"
# Install stanze
pip install "git+https://git@github.com/asyml/forte-wrappers.git#egg=forte.stanza&subdirectory=src/stanza"
# Install stave
pip install stave
```

Try starting the script with the following:

```bash 
python visualize.py
```

(After a ton of downloading) A web browser will pop up to show the results. You can enter sentence in the terminal prompt.
