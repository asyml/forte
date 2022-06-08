from testbook import testbook
import os

@testbook(
    "docs/notebook_tutorial/tutorial_MT_with_forte.ipynb", execute=False
)
def test_wrap_MT_inference_pipeline(tb):
    # if we just want to run through the notebook
    tb.execute_cell("pip_install")
    tb.execute_cell("def_pipeline")
    tb.execute_cell("terminal_reader")
    tb.execute_cell("add_sent_segmenter")
    # test Article ontology
    tb.execute_cell("def_article")
    tb.execute_cell("import_sent")
    tb.execute_cell("example_article")
    
    
    # test machine translation examples
    tb.execute_cell("def_mt_processor")
    tb.execute_cell("example_mt")
    tb.execute_cell("def_mt_mpprocessor")
    tb.execute_cell("example_mpmt")
    tb.execute_cell("example_html_mtmp")
    tb.execute_cell("def_html_tag_processor")
    tb.execute_cell("example_html_tag")
    tb.execute_cell("def_online_mt_processor")

    
    # test pipeline save and load
    tb.execute_cell("example_pipeline_save")
    tb.execute_cell("example_pipeline_load")
