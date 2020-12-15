# Copyright 2020 The Forte Authors. All Rights Reserved.
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
This file implements the UDA(Unsupervised Data Augmentation)
algorithm as described in
Unsupervised Data Augmentation for Consistency Training
(https://arxiv.org/abs/1904.12848)
(https://github.com/google-research/uda)
"""
from typing import Optional, List, Tuple, Iterator
from torch import Tensor
from texar.torch.losses.info_loss import kl_divg_loss_with_logits
from texar.torch.data import DataIterator
from texar.torch.data.data.dataset_utils import Batch


class UDAIterator:
    r"""
    This iterator wraps the Unsupervised Data Augmentation(UDA)
    algorithm by calculating the unsupervised loss automatically
    during each iteration. It takes both supervised and unsupervised
    data iterator as input.

    The unsupervised data should contain the original input and the
    augmented input. The original and augmented inputs should be in
    the same training example.

    During each iteration, the iterator will return the supervised and
    unsupervised batches. Users can call the :func:`calculate_uda_loss`
    to get the UDA loss and combine it with the supervised loss
    for model training.

    It uses tricks such as prediction sharpening and confidence masking.
    Please refer to the UDA paper for more details.
    (https://arxiv.org/abs/1904.12848)

    Args:
        sup_iterator: The iterator for supervised data. Each item is a
            training/eval/test example with key-value pairs as inputs.
        unsup_iterator: The iterator for unsupervised data. Each training
            example in it should contain both the original and augmented data.
        softmax_temperature: The softmax temperature for sharpening the
            distribution. The value should be larger than 0. Defaults to 1.
        confidence_threshold: The threshold for confidence-masking.
            It is a threshold of the probability in [0, 1],
            rather than of the logit. If set to -1, the threshold
            will be ignored. Defaults to -1.
        reduction: Default: 'mean'. This is the same as the `reduction`
            argument in
            :func:`texar.torch.losses.info_loss.kl_divg_loss_with_logits`.
            The loss will be a scalar tensor if the `reduction`
            is not :attr:`'none'`.
            Specifies the reduction to apply to the output:

            - ``'none'``: no reduction will be applied.
            - ``'batchmean'``: the sum of the output will be divided
              by the batchsize.
            - ``'sum'``: the output will be summed.
            - ``'mean'``: the output will be divided by the number of elements
              in the output.
    """
    def __init__(
            self,
            sup_iterator: DataIterator,
            unsup_iterator: DataIterator,
            softmax_temperature: float = 1.0,
            confidence_threshold: float = -1,
            reduction: str = "mean",
    ):
        self._sup_iterator: DataIterator = sup_iterator
        self._unsup_iterator: DataIterator = unsup_iterator
        self._softmax_temperature = softmax_temperature
        self._confidence_threshold = confidence_threshold
        self._reduction = reduction

        # The flag for returning the unsupervised data.
        self._use_unsup = True
        self._sup_iter: Iterator[Batch]
        self._unsup_iter: Iterator[Batch]

    def __len__(self):
        return self._sup_iterator.__len__()

    def switch_to_dataset(
            self,
            dataset_name: Optional[str] = None,
            use_unsup: bool = True
    ):
        # Set the flag of using unsupervised data.
        self._use_unsup = use_unsup
        self._sup_iterator.switch_to_dataset(dataset_name)

    def switch_to_dataset_unsup(
            self,
            dataset_name: Optional[str] = None
    ):
        self._unsup_iterator.switch_to_dataset(dataset_name)

    @property
    def num_datasets(self) -> int:
        return self._sup_iterator.num_datasets \
               + self._unsup_iterator.num_datasets

    @property
    def dataset_names(self) -> List[str]:
        return self._sup_iterator.dataset_names \
               + self._unsup_iterator.dataset_names

    def calculate_uda_loss(
            self,
            logits_orig: Tensor,
            logits_aug: Tensor
    ) -> Tensor:
        r"""
        This function calculate the KL divergence between the output
        probabilities of original input and augmented input. The two inputs
        should have the same shape, and the last dimension of them should be
        the probability distribution.

        Args:
            logits_orig: A tensor contains the logits of the original data.
            logits_aug: A tensor contains the logits of the augmented data.
                Must have the same shape as `logits_orig`.
        Returns:
            The loss, as a pytorch scalar float tensor if the `reduction`
            is not :attr:`'none'`, otherwise a tensor with the same shape
            as the `logits_orig`.
        """
        uda_loss = kl_divg_loss_with_logits(
            target_logits=logits_orig,
            input_logits=logits_aug,
            softmax_temperature=self._softmax_temperature,
            confidence_threshold=self._confidence_threshold,
            reduction=self._reduction
        )
        return uda_loss

    def __iter__(self):
        r"""
        The :class:`texar.torch.data.DataIterator` is not
        inherited from the :class:`Iterator`. So we have to
        get the iterator explicitly.
        """
        self._sup_iter = iter(self._sup_iterator)
        self._unsup_iter = iter(self._unsup_iterator)
        return self

    def __next__(self) -> Tuple[Batch, Optional[Batch]]:
        r"""
        When :attr:`_use_unsup` = False, the iterator will be the same
        as a normal iterator for the supervised data. Otherwise,
        it will yield unsupervised batch, in addition to
        the supervised batch.

        The iterator will only raise the StopIteration when
        the supervised dataset reaches its end. The unsupervised
        data will always be yielded as long as the iterator
        still has supervised data.
        """
        if not self._use_unsup:
            return next(self._sup_iter), None
        try:
            sup_batch = next(self._sup_iter)
        except StopIteration as e:
            raise StopIteration from e
        try:
            unsup_batch = next(self._unsup_iter)
        except StopIteration:
            self._unsup_iter = iter(self._unsup_iterator)
            unsup_batch = next(self._unsup_iter)

        return sup_batch, unsup_batch
