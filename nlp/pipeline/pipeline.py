from nlp.pipeline.processors.base_processor import BaseProcessor

class Pipeline:
    def __init__(self):
        self.reader = None
        # list of BaseProcessor
        self.processors = []

        self.current_packs = []

    def read_next(self):
        pass

    def process_next(self):
        for processor in self.processors:
            for pack in self.current_packs:
                processor.process(pack)
