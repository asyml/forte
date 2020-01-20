# ***automatically_generated***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated file. Do not change manually.
"""
import forte.data.data_pack
import forte.data.ontology.top
import ft.onto
import typing


__all__ = []


__all__.extend('Pet')


class Pet(forte.data.ontology.top.Annotation):
    """
    Pets in the shop.
    """

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._pet_type: typing.Optional[str] = None

    @property
    def pet_type(self):
        return self._pet_type

    def set_pet_type(self, pet_type: typing.Optional[str]):
        self.set_fields(_pet_type=pet_type)


__all__.extend('Owner')


class Owner(forte.data.ontology.top.Annotation):
    """
    Owner of pets.
        Args:
            pets (typing.Optional[typing.List[ft.onto.pet_shop.Pet]]): List of pets one can have.
    """

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._name: typing.Optional[str] = None
        self._pets: typing.Optional[typing.List[ft.onto.pet_shop.Pet]] = None

    @property
    def name(self):
        return self._name

    def set_name(self, name: typing.Optional[str]):
        self.set_fields(_name=name)

    @property
    def pets(self):
        return self._pets

    def set_pets(self, pets: typing.Optional[typing.List[ft.onto.pet_shop.Pet]]):
        self.set_fields(_pets=[item.tid for item in pets])
