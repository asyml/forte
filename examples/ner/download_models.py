from forte.data.data_utils import maybe_download

# download resources
urls = [
    "https://drive.google.com/file/d/1j3i5U1YffYqKTdSbnlsrMAm9j86nLjxC/view"
    "?usp=sharing",
    "https://drive.google.com/file/d/1aRPS_b4AFaZTsk9uZ41tkWIBvWbO_s_V/"
    "view?usp=sharing",
    "https://drive.google.com/file/d/1SYpcWEDeTlbOsXlRevS8YS_dyP_k29g0/"
    "view?usp=sharing",
    "https://drive.google.com/file/d/1S2UMDBX7Ci-Mrm30434t0LOBL__Db92Y/"
    "view?usp=sharing",
    "https://drive.google.com/file/d/1O4iFhBPuogwEgz7bpJjEqDqAlYf5caP4/"
    "view?usp=sharing",
]

filenames = [
    "model.pkl",
    "word_embedding_table.pkl",
    "word_alphabet.pkl",
    "ner_alphabet.pkl",
    "char_alphabet.pkl",
]

maybe_download(urls=urls, path="resources/", filenames=filenames)
