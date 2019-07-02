"""
The Resources class wraps necessary resources to build a processor ( or a
trainer)
"""


class Resources:
    def __init__(self, **kwargs):
        self.resources = kwargs
