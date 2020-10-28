import os
import sys

def main(arguments):
    import subprocess
    if not os.path.exists("data/IMDB_raw"):
        subprocess.run("mkdir data/IMDB_raw", shell=True)
    # pylint: disable=line-too-long
    subprocess.run(
        'wget -P data/IMDB_raw/ https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
        shell=True)
    subprocess.run(
        'tar xzvf data/IMDB_raw/aclImdb_v1.tar.gz -C data/IMDB_raw/ && rm data/IMDB_raw/aclImdb_v1.tar.gz',
        shell=True)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
