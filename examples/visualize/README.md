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

Try the following:

```bash 
python visualize.py
```

A web brow