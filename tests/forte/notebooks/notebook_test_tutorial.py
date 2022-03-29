from testbook import testbook


@testbook("docs/notebook_tutorial/handling_structued_data.ipynb")
def test_handling_structued_data_cells(tb):
    # test specific cells output
    # note that Developer needs to assign tags to cells
    # Please refer to https://jupyterbook.org/content/metadata.html for more infomation
    tb.execute_cell("imports")
    get_document_data_text = """0 :   The Indonesian billionaire James Riady has agreed to pay $ 8.5 million and plead guilty to illegally donating money for Bill Clinton 's 1992 presidential campaign . He admits he was trying to influence American policy on China ."""
    get_document_text = """0 document instance:   Document(document_class=[], sentiment={}, classifications=<forte.data.ontology.core.FDict object at 0x7f0654e37a50>)
0 document text:   The Indonesian billionaire James Riady has agreed to pay $ 8.5 million and plead guilty to illegally donating money for Bill Clinton 's 1992 presidential campaign . He admits he was trying to influence American policy on China ."""
    print(tb.execute_cell(1))
    assert tb.cell_output_text("get_document_data") == get_document_data_text

    assert tb.cell_output_text("get_document") == get_document_text


@testbook("docs/notebook_tutorial/handling_structued_data.ipynb", execute=True)
def test_handling_structued_data(tb):
    # if we simply want to run through the notebook
    pass


@testbook("docs/notebook_tutorial/notebook_test_tutorial.ipynb", execute=True)
def test_func(tb):
    # if we want to test a specific function defined in the notebook
    func = tb.ref("func")
    assert func(1, 2) == 3

    # Another way to testing is
    # Inject assertion into notebook
    tb.inject("assert func(1, 2) == 3")

    # we can also inject variable definition
    # such that we can skip running some cells
    tb.inject("a = 4")
    tb.inject("b = 3")
    tb.inject("assert func(a,b)==7")
