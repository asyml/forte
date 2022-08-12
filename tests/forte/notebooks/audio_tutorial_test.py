from testbook import testbook


@testbook(
    "docs/notebook_tutorial/Automatic_Speech_Recognition.ipynb", execute=True
)
def test_text_classification_pipeline(tb):
    # if we just want to run through the notebook
    pass