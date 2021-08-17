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
import argparse
import functools
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics.pairwise import paired_cosine_distances
import texar.torch as tx

import config_data

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", default="data/", help="Data directory to read the files from"
)
parser.add_argument(
    "--output_dir",
    default="model/",
    help="Output directory to write the pickled files",
)
args = parser.parse_args()


def get_lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    r"""Calculate the learning rate multiplier given current step and the number
    of warm-up steps. The learning rate schedule follows a linear warm-up and
    linear decay.
    """

    step = min(step, total_steps)

    multiplier = 1 - (step - warmup_steps) / (total_steps - warmup_steps)

    if warmup_steps > 0 and step < warmup_steps:
        warmup_percent_done = step / warmup_steps
        multiplier = warmup_percent_done

    return multiplier


class SiameseBert(nn.Module):
    r"""BERT model with siamese network structure. This class finetunes two
    BERT models with their weights tied on sentence similarity tasks.

    Args:
        num_classes (int): Number of labels for the classification task
        torch_device (torch.device): A device object which specifies the model
            placement.
    """

    def __init__(
        self, num_classes: int, torch_device: Optional[torch.device] = None
    ):
        super().__init__()
        self.bert = tx.modules.BERTEncoder(
            pretrained_model_name="bert-base-uncased"
        )
        self.bert.to(device=torch_device)
        self.num_classes = num_classes
        self.classifier = nn.Linear(
            in_features=3 * self.bert.output_size, out_features=num_classes
        )
        self.classifier.to(device=torch_device)

    def forward(
        self,
        sent_a_input_ids,
        sent_a_seq_len,
        sent_a_segment_ids,
        sent_b_input_ids,
        sent_b_seq_len,
        sent_b_segment_ids,
    ):
        output, _ = self.bert(
            inputs=sent_a_input_ids,
            sequence_length=sent_a_seq_len,
            segment_ids=sent_a_segment_ids,
        )
        sent_a_embedding = output[:, 0, :]

        output, _ = self.bert(
            inputs=sent_b_input_ids,
            sequence_length=sent_b_seq_len,
            segment_ids=sent_b_segment_ids,
        )
        sent_b_embedding = output[:, 0, :]

        vectors = [
            sent_a_embedding,
            sent_b_embedding,
            torch.abs(sent_a_embedding - sent_b_embedding),
        ]
        vectors = torch.cat(vectors, dim=1)

        logits = self.classifier(vectors)
        preds = torch.argmax(logits, dim=-1)

        return sent_a_embedding, sent_b_embedding, logits, preds


def _compute_loss(logits, labels):
    r"""Compute loss."""

    loss = F.cross_entropy(
        logits.view(-1, chatbot_bert.num_classes),
        labels.view(-1),
        reduction="mean",
    )
    return loss


def _train_epoch():
    r"""Trains on the training set, and evaluates on the dev set
    periodically.
    """

    data_iterator.switch_to_dataset("train")
    chatbot_bert.train()

    for batch in data_iterator:
        optim.zero_grad()
        _, _, logits, _ = chatbot_bert(
            sent_a_input_ids=batch["sent_a_input_ids"],
            sent_a_seq_len=batch["sent_a_seq_len"],
            sent_a_segment_ids=batch["sent_a_segment_ids"],
            sent_b_input_ids=batch["sent_b_input_ids"],
            sent_b_seq_len=batch["sent_b_seq_len"],
            sent_b_segment_ids=batch["sent_b_segment_ids"],
        )
        labels = batch["label_ids"]

        loss = _compute_loss(logits, labels)
        loss.backward()
        optim.step()
        scheduler.step()
        step = scheduler.last_epoch
        step += 1

        dis_steps = config_data.display_steps
        if dis_steps > 0 and step % dis_steps == 0:
            print(f"step: {step}; loss: {loss.item()}")

        eval_steps = config_data.eval_steps
        if eval_steps > 0 and step % eval_steps == 0:
            _eval_epoch(dataset="eval")


@torch.no_grad()
def _eval_epoch(dataset="eval"):
    r"""Evaluates on ``dataset``."""

    data_iterator.switch_to_dataset(dataset)
    chatbot_bert.eval()

    embeddings_a = []
    embeddings_b = []
    labels = []
    nsamples = 0
    avg_rec = tx.utils.AverageRecorder()
    for batch in data_iterator:
        sent_a_embedding, sent_b_embedding, _, preds = chatbot_bert(
            sent_a_input_ids=batch["sent_a_input_ids"],
            sent_a_seq_len=batch["sent_a_seq_len"],
            sent_a_segment_ids=batch["sent_a_segment_ids"],
            sent_b_input_ids=batch["sent_b_input_ids"],
            sent_b_seq_len=batch["sent_b_seq_len"],
            sent_b_segment_ids=batch["sent_b_segment_ids"],
        )

        label_ids = batch["label_ids"]
        labels.extend(label_ids.to("cpu").numpy())
        embeddings_a.extend(sent_a_embedding.to("cpu").numpy())
        embeddings_b.extend(sent_b_embedding.to("cpu").numpy())

        accu = tx.evals.accuracy(label_ids, preds)
        avg_rec.add([accu], batch["sent_a_input_ids"].size(1))
        nsamples += len(batch)

    cosine_scores = 1 - (paired_cosine_distances(embeddings_a, embeddings_b))
    threshold = 0.5
    predictions = np.array(cosine_scores > threshold, dtype=int)
    cosine_accuracy = np.sum(predictions == labels) / len(labels)
    print(
        f"Evaluating on {dataset} dataset."
        f"Accuracy based on Cosine Similarity: {cosine_accuracy},"
        f"Accuracy based on logits: {avg_rec.avg(0)},"
        f"nsamples: {nsamples}"
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chatbot_bert = SiameseBert(num_classes=2, torch_device=device)

    num_train_data = config_data.num_train_data
    num_train_steps = int(
        num_train_data
        / config_data.train_batch_size
        * config_data.max_train_epoch
    )
    num_warmup_steps = int(num_train_steps * config_data.warmup_proportion)

    # Builds learning rate decay scheduler
    static_lr = 2e-5

    vars_with_decay = []
    vars_without_decay = []
    for name, param in chatbot_bert.named_parameters():
        if "layer_norm" in name or name.endswith("bias"):
            vars_without_decay.append(param)
        else:
            vars_with_decay.append(param)

    opt_params = [
        {
            "params": vars_with_decay,
            "weight_decay": 0.01,
        },
        {
            "params": vars_without_decay,
            "weight_decay": 0.0,
        },
    ]
    optim = tx.core.BertAdam(
        opt_params, betas=(0.9, 0.999), eps=1e-6, lr=static_lr
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        functools.partial(
            get_lr_multiplier,
            total_steps=num_train_steps,
            warmup_steps=num_warmup_steps,
        ),
    )

    train_dataset = tx.data.RecordData(
        hparams=config_data.train_hparam, device=device
    )
    eval_dataset = tx.data.RecordData(
        hparams=config_data.eval_hparam, device=device
    )
    test_dataset = tx.data.RecordData(
        hparams=config_data.test_hparam, device=device
    )

    data_iterator = tx.data.DataIterator(
        {"train": train_dataset, "eval": eval_dataset, "test": test_dataset}
    )

    for _ in range(config_data.max_train_epoch):
        print("Finetuning BERT for chatbot...")
        _train_epoch()

    _eval_epoch(dataset="test")

    print("Saving the model...")
    states = {
        "bert": chatbot_bert.bert.state_dict(),
        "classifier": chatbot_bert.classifier.state_dict(),
    }
    with open(Path(args.output_dir, "chatbot_model.ckpt"), "wb") as f:
        pickle.dump(states, f)
