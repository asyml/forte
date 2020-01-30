# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from forte.data import DataPack
from forte.data.ontology import Annotation, Generics
from ft.onto.base_ontology import Document


class WikiPage(Document):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._body: WikiBody
        self._page_id: str
        self._page_name: str

    def set_page_id(self, pid: str):
        self.set_fields(_page_id=pid)

    @property
    def page_id(self):
        return self._page_id

    def set_page_name(self, page_name: str):
        self.set_fields(_page_name=page_name)

    def page_name(self):
        return self._page_name


# pylint: disable=useless-super-delegation
class WikiBody(Annotation):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class WikiSection(Annotation):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._is_intro: bool

    def set_is_intro(self, is_intro: bool):
        self.set_fields(_is_intro=is_intro)

    @property
    def is_intro(self):
        return self._is_intro


class WikiParagraph(Annotation):
    pass


class WikiTitle(Annotation):
    pass


class WikiAnchor(Annotation):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._target_page_name: str

    @property
    def target_page_name(self):
        return self._target_page_name

    def set_target_page_name(self, page_name: str):
        self.set_fields(_target_page_name=page_name)


class WikiInfoBoxProperty(Generics):
    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self._key: str
        self._value: str

    @property
    def key(self):
        return self._key

    def set_key(self, key: str):
        self.set_fields(_key=key)

    @property
    def value(self):
        return self._value

    def set_value(self, value: str):
        self.set_fields(_value=value)


class WikiInfoBoxMapped(Generics):
    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self._key: str
        self._value: str
        self._infobox_type: str

    @property
    def key(self):
        return self._key

    def set_key(self, key: str):
        self.set_fields(_key=key)

    @property
    def value(self):
        return self._value

    def set_value(self, value: str):
        self.set_fields(_value=value)

    @property
    def infobox_type(self):
        return self._infobox_type

    def set_infobox_type(self, infobox_type: str):
        self.set_fields(_infobox_type=infobox_type)
