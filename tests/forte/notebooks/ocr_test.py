from testbook import testbook
import os

@testbook(
    "docs/notebook_tutorial/ocr.ipynb", execute=False
)
def test_wrap_MT_inference_pipeline(tb):
    # if we just want to run through the notebook
    tb.execute_cell("ocr_reader")
    tb.execute_cell("ocr_char_processor")
    tb.execute_cell("ocr_token_processor")
    tb.execute_cell("get_image")
    tb.execute_cell("pipeline")
    tb.execute_cell("recognize_char")
    tb.execute_cell("recognize_token")
