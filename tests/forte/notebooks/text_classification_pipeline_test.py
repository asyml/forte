from testbook import testbook


@testbook(
    "docs/notebook_tutorial/zero_shot_text_classification.ipynb", execute=False
)
def test_text_classification_pipeline(tb):
    # if we just want to run through the notebook
    pass
