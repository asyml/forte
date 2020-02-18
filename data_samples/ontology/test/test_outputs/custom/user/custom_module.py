# ***automatically_generated***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated file. Do not change manually.
"""
import custom.user
import forte.data.data_pack
import forte.data.ontology.top
import ft.onto
import typing


__all__ = []


__all__.extend('Dependency')


class Dependency(forte.data.ontology.top.Link):
    parent_type: ft.onto.ft_module.Token = None
    child_type: ft.onto.ft_module.Token = None

    def __init__(self, pack: forte.data.base_pack.PackType, parent: typing.Optional[forte.data.ontology.core.Entry] = None, child: typing.Optional[forte.data.ontology.core.Entry] = None):
        super().__init__(pack, parent, child)
        self._rel_type: typing.Optional[str] = None

    @property
    def rel_type(self):
        return self._rel_type

    def set_rel_type(self, rel_type: typing.Optional[str]):
        self.set_fields(_rel_type=rel_type)
