from pathlib import Path

from forte.data.data_utils import maybe_download

if __name__ == "__main__":
    model_path = Path("examples/Cliner/CliNER/models")
    pkg_path = Path("examples/Cliner/CliNER/tools")

    # create the model path if it doesn't exist
    model_path.mkdir(parents=True, exist_ok=True)

    # download the pre-trained model
    maybe_download(urls=[
        "https://drive.google.com/file/d/1Jlm2wdmNA-GotTWF60zZRUs1MbnzYox2"],
        path=model_path,
        filenames=["train_full.model"])

    # download the dependency package for evaluation
    maybe_download(urls=[
        "https://drive.google.com/file/d/1ZVgJ7EQtMjPpg_v-lCycCLdFgVzmdTxI"],
        path=pkg_path,
        filenames=["i2b2va-eval.jar"])

    maybe_download(urls=[
        "https://drive.google.com/file/d/1QvenOvRx7R9XjA5tUxeZnDdoroW94R-N"
        "/view?usp=sharing"],
        path=pkg_path,
        filenames=["py3_maxent_treebank_pos_tagger.pickle"])

    maybe_download(urls=[
        "https://drive.google.com/file/d/162jCbpzf5Jez8h0zzA-lAyMkCLcjgAep"
        "/view?usp=sharing"],
        path=pkg_path,
        filenames=["py2_maxent_treebank_pos_tagger.pickle"])
