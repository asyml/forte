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


__all__.extend('WikiBody')


class WikiBody(forte.data.ontology.top.Annotation):
    """
    Entry defining a phrase in the document.
    """

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)


__all__.extend('WikiPage')


class WikiPage(forte.data.ontology.top.Annotation):

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._body: typing.Optional[ft.onto.wikipedia.WikiBody] = None
        self._page_id: typing.Optional[str] = None
        self._page_name: typing.Optional[str] = None

    @property
    def body(self):
        return self._body

    def set_body(self, body: typing.Optional[ft.onto.wikipedia.WikiBody]):
        self.set_fields(_body=body.tid)

    @property
    def page_id(self):
        return self._page_id

    def set_page_id(self, page_id: typing.Optional[str]):
        self.set_fields(_page_id=page_id)

    @property
    def page_name(self):
        return self._page_name

    def set_page_name(self, page_name: typing.Optional[str]):
        self.set_fields(_page_name=page_name)


__all__.extend('WikiParagraph')


class WikiParagraph(forte.data.ontology.top.Annotation):
    """
    a paragraph in the document.
    """

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)


__all__.extend('WikiTitle')


class WikiTitle(forte.data.ontology.top.Annotation):
    """
    the title of the document.
    """

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)


__all__.extend('WikiAnchor')


class WikiAnchor(forte.data.ontology.top.Annotation):
    """
    an anchor is a text field that link to other part of the document.
    """

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._target_page_name: typing.Optional[str] = None

    @property
    def target_page_name(self):
        return self._target_page_name

    def set_target_page_name(self, target_page_name: typing.Optional[str]):
        self.set_fields(_target_page_name=target_page_name)


__all__.extend('WikiInfoBoxProperty')


class WikiInfoBoxProperty(forte.data.ontology.top.Generic):
    """
    represents one info box in the property name space of the page.
    """

    def __init__(self, pack: forte.data.base_pack.PackType):
        super().__init__(pack)
        self._key: typing.Optional[str] = None
        self._value: typing.Optional[str] = None

    @property
    def key(self):
        return self._key

    def set_key(self, key: typing.Optional[str]):
        self.set_fields(_key=key)

    @property
    def value(self):
        return self._value

    def set_value(self, value: typing.Optional[str]):
        self.set_fields(_value=value)


__all__.extend('WikiInfoBoxMapped')


class WikiInfoBoxMapped(forte.data.ontology.top.Generic):
    """
    represents one info box in the mapped namespace of the page.
    """

    def __init__(self, pack: forte.data.base_pack.PackType):
        super().__init__(pack)
        self._key: typing.Optional[str] = None
        self._value: typing.Optional[str] = None
        self._infobox_type: typing.Optional[str] = None

    @property
    def key(self):
        return self._key

    def set_key(self, key: typing.Optional[str]):
        self.set_fields(_key=key)

    @property
    def value(self):
        return self._value

    def set_value(self, value: typing.Optional[str]):
        self.set_fields(_value=value)

    @property
    def infobox_type(self):
        return self._infobox_type

    def set_infobox_type(self, infobox_type: typing.Optional[str]):
        self.set_fields(_infobox_type=infobox_type)
