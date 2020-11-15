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
from typing import Optional, List, Callable, Tuple
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
    the same training example. The user should provide a function
    unsup_forward_fn to calculate the output logits for original &
    augmented inputs from the unsupervised batch.

    During each iteration, the iterator will calculate the UDA loss
    automatically with the given unsup_forward_fn. User can combine the
    UDA loss with the supervised loss for model training.

    It uses tricks such as prediction sharpening and confidence masking.
    Please refer to the UDA paper for more details.
    (https://arxiv.org/abs/1904.12848)

    Args:
        - sup_iterator: The iterator for supervised data.
        - unsup_iterator:
            The iterator for unsupervised data. Each example in it
            should contain both the original and augmented data.
        - unsup_forward_fn:
            The function to calculate the UDA loss. The signature should be:
                unsup_forward_fn(unsup_batch: Batch) -> Tuple[Tensor, Tensor]
            The return values are [original_logits, augmented_logits].
            The last dimension of the logits should be the distribution of
            prediction classes.
        - softmax_temperature:
            The softmax temperature for sharpening the distribution.
        - confidence_threshold:
            The threshold for confidence-masking. It is a threshold
            of the probability in [0, 1], rather than of the logit.
            If set to -1, the threshold will be ignored.
        - reduction:
            Default: 'mean'.
            This is the same as the `reduction` argument in :func:
            texar.losses.info_loss.kl_divg_loss_with_logits.
            Specifies the reduction to apply to the output:

            - :attr:`'none'`: no reduction will be applied.
            - :attr:`'batchmean'`: the sum of the output will be
                divided by the batchsize.
            - :attr:`'sum'`: the output will be summed.
            - :attr:`'mean'`: the output will be divided by
                the number of elements in the output.
    """
    def __init__(
            self,
            sup_iterator: DataIterator,
            unsup_iterator: DataIterator,
            unsup_forward_fn: Callable,
            softmax_temperature: float = 1.0,
            confidence_threshold: float = -1,
            reduction: str = "mean",
    ):
        self.sup_iterator: DataIterator = sup_iterator
        self.unsup_iterator: DataIterator = unsup_iterator
        self.unsup_forward_fn = unsup_forward_fn
        self.softmax_temperature = softmax_temperature
        self.confidence_threshold = confidence_threshold
        self.reduction = reduction

        # The flag for returning the unsupervised data.
        self.use_unsup = True
        self.sup_iter: Optional[DataIterator[Batch]] = None
        self.unsup_iter: Optional[DataIterator[Batch]] = None

    def __len__(self):
        return self.sup_iterator.__len__()

    def switch_to_dataset(
            self,
            dataset_name: Optional[str] = None,
            use_unsup: bool = True
    ):
        # Set the flag of using unsupervised data.
        self.use_unsup = use_unsup
        self.sup_iterator.switch_to_dataset(dataset_name)

    def switch_to_dataset_unsup(
            self,
            dataset_name: Optional[str] = None
    ):
        self.unsup_iterator.switch_to_dataset(dataset_name)

    @property
    def num_datasets(self) -> int:
        return self.sup_iterator.num_datasets \
               + self.unsup_iterator.num_datasets

    @property
    def dataset_names(self) -> List[str]:
        return self.sup_iterator.dataset_names \
               + self.unsup_iterator.dataset_names

    def calculate_uda_loss(
            self,
            unsup_batch: Batch,
            unsup_forward_fn: Callable
    ) -> Tensor:
        r"""
        This function calculate the KL divergence
        between the output probabilities of original
        input and augmented input. It calls the user-provided
        :func: unsup_forward_fn to perform the prediction.

        Args:
            - unsup_batch: A batch containing the unsupervised data.
                There should be both original & augmented inputs in
                each training example.
            - unsup_forward_fn: The user-provided forward function
                of the model.
        Returns:
            - The loss, as a pytorch scalar float tensor.
        """
        logits_orig, logits_aug = unsup_forward_fn(unsup_batch)
        uda_loss = kl_divg_loss_with_logits(
            target_logits=logits_orig,
            input_logits=logits_aug,
            softmax_temperature=self.softmax_temperature,
            confidence_threshold=self.confidence_threshold,
            reduction=self.reduction
        )
        return uda_loss

    def __iter__(self):
        self.sup_iter = iter(self.sup_iterator)
        self.unsup_iter = iter(self.unsup_iterator)
        return self

    def __next__(self) -> Tuple[Batch, Optional[Batch], Optional[Tensor]]:
        r"""
        When use_unsup = False, the iterator will be the same
        as a normal iterator for the supervised data. Otherwise,
        it will yield unsupervised batch & loss, in addition to
        the supervised batch.

        The iterator will only raise the StopIteration when
        the supervised dataset reaches its end. The unsupervised
        data will always be yielded as long as the iterator
        still has supervised data.

        It will calculate the UDA loss for the unsupervised data,
        and yield it along with the supervised & unsupervised batch.

        """
        if not self.use_unsup:
            return next(self.sup_iter), None, None
        try:
            sup_batch = next(self.sup_iter)
        except StopIteration:
            raise StopIteration
        try:
            unsup_batch = next(self.unsup_iter)
        except StopIteration:
            self.unsup_iter = iter(self.unsup_iterator)
            unsup_batch = next(self.unsup_iter)

        unsup_loss = self.calculate_uda_loss(unsup_batch, self.unsup_forward_fn)
        return sup_batch, unsup_batch, unsup_loss
