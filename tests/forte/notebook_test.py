from testbook import testbook


@testbook("docs/notebook_tutorial/pipeline.ipynb", execute=True)
def test_get_details(tb):
    assert True


@testbook("docs/notebook_tutorial/handling_structued_data.ipynb", execute=True)
def test_get_details(tb):
    assert True


@testbook(
    "docs/notebook_tutorial/text_classification_pipeline.ipynb", execute=True
)
def test_get_details(tb):
    assert True


@testbook(
    "docs/notebook_tutorial/wrap_MT_inference_pipeline.ipynb", execute=True
)
def test_get_details(tb):
    assert True
