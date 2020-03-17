# ***automatically_generated***
# ***source json:forte/ontology_specs/wikipedia_ontology.json***
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
        _page_name (Optional[str])
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._page_id: Optional[str] = None
        self._page_name: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['page_id'] = state.pop('_page_id')
        state['page_name'] = state.pop('_page_name')
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self._page_id = state.get('page_id', None) 
        self._page_name = state.get('page_name', None) 

    @property
    def page_id(self):
        return self._page_id

    @page_id.setter
    def page_id(self, page_id: Optional[str]):
        self.set_fields(_page_id=page_id)

    @property
    def page_name(self):
        return self._page_name

    @page_name.setter
    def page_name(self, page_name: Optional[str]):
        self.set_fields(_page_name=page_name)


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
        state['is_intro'] = state.pop('_is_intro')
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
        state['target_page_name'] = state.pop('_target_page_name')
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
        state['key'] = state.pop('_key')
        state['value'] = state.pop('_value')
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
        state['key'] = state.pop('_key')
        state['value'] = state.pop('_value')
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
