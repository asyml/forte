from forte.data.data_utils import maybe_download

# download resources
urls = ["https://drive.google.com/file/d/15RSfFkW9syQKtx-_fQ9KshN3BJ27Jf8t/"
        "view?usp=sharing",
        "https://drive.google.com/file/d/1Nh7D6Xam5JefdoSXRoL7S0DZK1d4i2UK/"
        "view?usp=sharing",
        "https://drive.google.com/file/d/1YWcI60lGKtTFH01Ai1HnwOKBsrFf2r29/"
        "view?usp=sharing",
        "https://drive.google.com/file/d/1ElHUEMPQIuWmV0GimroqFphbCvFKskYj/"
        "view?usp=sharing",
        "https://drive.google.com/file/d/1EhMXlieoEg-bGUbbQ2vN-iyNJvC4Dajl/"
        "view?usp=sharing"]

filenames = ["config.json",
             "pytorch_model.bin",
             "special_tokens_map.json",
             "tokenizer_config.json",
             "vocab.txt"]

maybe_download(urls=urls, path="resources/NCBI-disease", filenames=filenames)
