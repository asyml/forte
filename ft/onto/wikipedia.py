# ***automatically_generated***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology wikipedia. Do not change manually.
"""

from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation
from forte.data.ontology.top import Generics
from typing import Optional


__all__ = [
    "WikiPage",
    "WikiBody",
    "WikiSection",
    "WikiParagraph",
    "WikiTitle",
    "WikiAnchor",
    "WikiInfoBoxProperty",
    "WikiInfoBoxMapped",
]


class WikiPage(Annotation):
    """
    Attributes:
        _page_id (Optional[str])
        _part_name (Optional[str])
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._page_id: Optional[str] = None
        self._part_name: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['page_id'] = self._page_id
        state['part_name'] = self._part_name
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self._page_id = state.get('page_id', None) 
        self._part_name = state.get('part_name', None) 

    @property
    def page_id(self):
        return self._page_id

    @page_id.setter
    def page_id(self, page_id: Optional[str]):
        self.set_fields(_page_id=page_id)

    @property
    def part_name(self):
        return self._part_name

    @part_name.setter
    def part_name(self, part_name: Optional[str]):
        self.set_fields(_part_name=part_name)


class WikiBody(Annotation):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class WikiSection(Annotation):
    """
    Attributes:
        _is_intro (Optional[bool])
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._is_intro: Optional[bool] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['is_intro'] = self._is_intro
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self._is_intro = state.get('is_intro', None) 

    @property
    def is_intro(self):
        return self._is_intro

    @is_intro.setter
    def is_intro(self, is_intro: Optional[bool]):
        self.set_fields(_is_intro=is_intro)


class WikiParagraph(Annotation):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class WikiTitle(Annotation):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class WikiAnchor(Annotation):
    """
    Attributes:
        _target_page_name (Optional[str])
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._target_page_name: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['target_page_name'] = self._target_page_name
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self._target_page_name = state.get('target_page_name', None) 

    @property
    def target_page_name(self):
        return self._target_page_name

    @target_page_name.setter
    def target_page_name(self, target_page_name: Optional[str]):
        self.set_fields(_target_page_name=target_page_name)


class WikiInfoBoxProperty(Generics):
    """
    Attributes:
        _key (Optional[str])
        _value (Optional[str])
    """
    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self._key: Optional[str] = None
        self._value: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['key'] = self._key
        state['value'] = self._value
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self._key = state.get('key', None) 
        self._value = state.get('value', None) 

    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, key: Optional[str]):
        self.set_fields(_key=key)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: Optional[str]):
        self.set_fields(_value=value)


class WikiInfoBoxMapped(Generics):
    """
    Attributes:
        _key (Optional[str])
        _value (Optional[str])
    """
    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self._key: Optional[str] = None
        self._value: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['key'] = self._key
        state['value'] = self._value
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self._key = state.get('key', None) 
        self._value = state.get('value', None) 

    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, key: Optional[str]):
        self.set_fields(_key=key)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: Optional[str]):
        self.set_fields(_value=value)
