from testbook import testbook

@testbook(
    "docs/notebook_tutorial/Automatic_Speech_Recognition.ipynb",
    execute=False
)
def test_Automatic_Speech_Recognition(tb):
    # input file
    tb.execute_cell("input_file")
    # install
    tb.execute_cell("install")
    tb.execute_cell("install2")
    # imports_1
    tb.execute_cell("imports_1")
    # processor 1
    tb.execute_cell("SpeakerSegmentation")
    tb.execute_cell("pipeline1")
    tb.execute_cell("import_2")
    tb.execute_cell("AudioUtterance")
    tb.execute_cell("pipeline2")
    tb.execute_cell("inference")
    