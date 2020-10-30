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
import tensorflow as tf

def kl_divergence_log_probs(log_p, log_q):
    r"""
    This function calculates the Kullback-Leibler Divergence
    between two distributions. The inputs are batched log probabilities.

    Args:
        - log_p: A batch of probabilities with a shape of [batch_size, dimension].
        - log_q: A batch of probabilities with a shape of [batch_size, dimension].
    Returns:
        The KL divergence for the probability pairs with a shape of [batch_size]
    """
    p = tf.exp(log_p)
    neg_ent = tf.reduce_sum(p * log_p, axis=-1)
    neg_cross_ent = tf.reduce_sum(p * log_q, axis=-1)
    kl = neg_ent - neg_cross_ent
    return kl

def calc_uda_loss(ori_log_probs, aug_log_probs, softmax_temp=-1, confidence_thres=-1):
    r"""
    This function calculates the unsupervised loss for UDA.
    The output probability distributions for original & augmented inputs
    are expected to be similar.

    It uses tricks like confidence-based masking and sharpening the distribution.
    Please refer to the original paper for more details.
    (https://arxiv.org/abs/1904.12848)
    Args:
        - ori_log_probs(shape: [batch_size, dim]):
            The output probabilities of original inputs.
        - aug_log_probs(shape: [batch_size, dim]):
            The output probabilities of augmented inputs.
        - softmax_temp: The softmax temparature for sharpening the distribution.
        - confidence_thres: The threshold for confidence-masking.
    Returns:
        The loss term in tensorflow.
    """
    unsup_loss_mask = 1
    # Sharpening the target distribution.
    if softmax_temp != -1:
        tgt_ori_log_probs = tf.nn.log_softmax(
            ori_log_probs / softmax_temp,
            axis=-1)
        tgt_ori_log_probs = tf.stop_gradient(tgt_ori_log_probs)
    else:
        tgt_ori_log_probs = tf.stop_gradient(ori_log_probs)
    # Mask the training sample based on confidence.
    if confidence_thres != -1:
        largest_prob = tf.reduce_max(tf.exp(ori_log_probs), axis=-1)
        unsup_loss_mask = tf.cast(tf.greater(
            largest_prob, confidence_thres), tf.float32)
        unsup_loss_mask = tf.stop_gradient(unsup_loss_mask)
    # Calculate the KL divergence.
    per_example_kl_loss = kl_divergence_log_probs(
        tgt_ori_log_probs, aug_log_probs) * unsup_loss_mask
    loss = tf.reduce_mean(per_example_kl_loss)
    return loss