# Copyright 2022 The Forte Authors. All Rights Reserved.
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

"""
SingleAnnotationAugmentOp is the extension of the BaseDataAugmentationOp.
The aim of this Op is to increase ease of use and reduce freedom of
operations that can be done on a data pack. This Op only allows the
implementation of those augmentation algorithms that process one
annotation at a time. This annotation can be one Token or even one
Sentence. But once declares, the augmentation will happen only
on the specified annotations.
"""
from typing import (
    Tuple,
    Dict,
    Any,
)
from abc import abstractmethod
from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation
from forte.utils.utils import get_class
from forte.processors.data_augment.algorithms.base_data_augmentation_op import (
    BaseDataAugmentationOp,
)


__all__ = ["SingleAnnotationAugmentOp"]


class SingleAnnotationAugmentOp(BaseDataAugmentationOp):
    r"""
    This class extends the `BaseDataAugmentationOp` to
    only allow augmentation of one annotation at a time.
    This operation should be used when we only want to
    augment one type of annotation in the whole data pack.
    Thus, to use this operation, the developer only needs
    to specify how a single annotation will be processed
    as a part of their augmentation method.
    We leave the :func:`single_annotation_augment` method to
    be implemented by the subclass. This function will specify
    what type of augmentation will a given annotation (of a
    predefined type) undergo.
    """

    def augment(self, data_pack: DataPack) -> bool:
        r"""
        This method is not to be modified when using
        the `SingleAnnotationAugmentOp`. This function takes
        in the augmentation logic specified by :func:`single_annotation_augment`
        method to apply it to each annotation of the specified type individually.

        Args:
            input_anno: the input annotation to be replaced.

        Returns:
            A boolean value indicating if the augmentation
            was successful (True) or unsuccessful (False).
        """
        augment_entry = get_class(self.configs["augment_entry"])
        anno: Annotation
        replaced_text: str
        is_replace: bool
        for anno in data_pack.get(augment_entry):
            is_replace, replaced_text = self.single_annotation_augment(anno)
            if is_replace:
                try:
                    _ = self.replace_annotations(anno, replaced_text)
                except ValueError:
                    return False
        return True

    @abstractmethod
    def single_annotation_augment(
        self, input_anno: Annotation
    ) -> Tuple[bool, str]:
        r"""
        This function takes in one annotation at a time and performs
        the desired augmentation on it. Through this function, one
        annotation is processed at a time. The developer needs to specify the
        logic that will be adopted to process one annotation of a given type.
        This method cannot suggest an augmentation logic which take in multiple
        annotations of the same type.

        Args:
            input_anno: The annotation that needs to be augmented.

        Returns:
            A tuple, where the first element is a boolean value indicating
            whether the augmentation happens, and the second element is the
            replaced string.
        """
        raise NotImplementedError

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        r"""
        Returns:
            A dictionary with the default config for this processor.
        Following are the keys for this dictionary:
            - augment_entry:
                Defines the entry the processor will augment.
                It should be a full qualified name of the entry class.
                Default value is "ft.onto.base_ontology.Token".
        """
        return {"augment_entry": "ft.onto.base_ontology.Token"}
